"""
NETRA — Video Perception Pipeline: ByteTrack + Plate Detection + License Plate OCR
File: tracking/plate_track_pipeline.py
Component: Checkpoint 4 — Video Multi-Object Tracking & Temporal OCR Aggregation
Project: SIH 2026 NETRA (Networked Engine for Traffic Recognition & Analytics)

Architecture:
    Video Stream
         ↓
    Vehicle Detection (YOLOv8n UVH-26)
         ↓
    ByteTrack Association (Kalman Filter + 2-Stage IoU)
         ↓
    Vehicle Track ID & Bounding Box
         ↓
    License Plate Detection (YOLOv8n Plate Detector, imgsz=1280)
         ↓
    Geometric Association (Plate Center ↔ Vehicle BBox Containment / IoU)
         ↓
    OCR Sampling (OCR every N frames per track)
         ↓
    PaddleOCR Recognition-Only (det=False, rec=True, cls=True) on CPU
         ↓
    Temporal Confidence-Weighted Voting & State Aggregation
         ↓
    Plate Stability Evaluation (stable / tentative / unknown)
         ↓
    Unified Vehicle Track Record + Annotated MP4 Video
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.license_plate_ocr import LicensePlateOCR

# Configure module logger
logger = logging.getLogger("NETRA.VideoPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [NETRA.Video] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default Model Paths
DEFAULT_VEHICLE_MODEL = "runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/best.pt"
DEFAULT_PLATE_MODEL = "runs/detect/runs/plate_detection/vehicle_plate_yolov8n_50ep/weights/best.pt"
DEFAULT_TRACKER_CONFIG = "tracking/tracker_config.yaml"

# Curated Palette for Vehicle Classes
CLASS_COLORS = {
    0: (230, 160, 40),   # car           - Ocean Blue / Cyan
    1: (60, 220, 90),    # motorcycle    - Emerald Green
    2: (0, 215, 255),    # auto_rickshaw - Indian Yellow / Gold
    3: (40, 70, 230),    # bus           - Crimson Red
    4: (220, 70, 220),   # truck         - Magenta / Purple
    5: (180, 200, 50),   # van           - Lime Turquoise
}
DEFAULT_VEHICLE_COLOR = (180, 180, 180)


def compute_box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class OCRObservation:
    """Single OCR observation recorded for a vehicle track."""
    frame_idx: int
    raw_text: str
    cleaned_text: str
    ocr_confidence: float
    valid_format: bool


class VehicleTrackState:
    """Maintains local vehicle track lifecycle, OCR observation history, and temporal voting."""

    def __init__(self, track_id: int, vehicle_class: str, first_seen_frame: int):
        self.track_id: int = int(track_id)
        self.vehicle_class: str = str(vehicle_class)
        self.first_seen_frame: int = int(first_seen_frame)
        self.last_seen_frame: int = int(first_seen_frame)
        self.last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.last_plate_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_ocr_frame: int = -999

        # Raw observation history
        self.observations: List[OCRObservation] = []

        # Aggregated voting attributes
        self.plate_text: str = ""
        self.plate_status: str = "unknown"  # "unknown" | "tentative" | "stable"
        self.plate_confidence: float = 0.0
        self.best_ocr_confidence: float = 0.0
        self.observation_count: int = 0
        self.valid_observation_count: int = 0

    def add_observation(
        self,
        obs: OCRObservation,
        min_valid_observations: int = 3,
        min_stability_score: float = 0.70,
    ) -> None:
        """Adds a new OCR observation and updates temporal confidence-weighted voting."""
        self.observations.append(obs)
        self.update_aggregation(
            min_valid_observations=min_valid_observations,
            min_stability_score=min_stability_score,
        )

    def update_aggregation(
        self,
        min_valid_observations: int = 3,
        min_stability_score: float = 0.70,
    ) -> None:
        """
        Executes confidence-weighted voting across all recorded observations.

        Rules:
        - Only observations with valid_format=True and confidence >= 0.50 contribute to primary vote
        - Each observation's vote is weighted by its OCR confidence
        - If winning candidate has >= MIN_VALID_OBSERVATIONS and plate_confidence >= MIN_STABILITY_SCORE:
            plate_status = "stable"
        - Elif at least 1 valid observation or confident candidate exists:
            plate_status = "tentative"
        - Else:
            plate_status = "unknown"
        """
        self.observation_count = len(self.observations)
        valid_obs = [
            obs for obs in self.observations
            if obs.valid_format and obs.ocr_confidence >= 0.50
        ]
        self.valid_observation_count = len(valid_obs)

        if valid_obs:
            # Accumulate confidence weights per candidate string
            vote_weights: Dict[str, float] = defaultdict(float)
            vote_counts: Dict[str, int] = defaultdict(int)

            for obs in valid_obs:
                vote_weights[obs.cleaned_text] += obs.ocr_confidence
                vote_counts[obs.cleaned_text] += 1

            # Determine candidate with strongest accumulated confidence
            winning_plate = max(vote_weights.keys(), key=lambda k: vote_weights[k])
            self.plate_text = winning_plate
            self.plate_confidence = round(
                vote_weights[winning_plate] / float(vote_counts[winning_plate]), 4
            )
            self.best_ocr_confidence = round(
                max(obs.ocr_confidence for obs in valid_obs if obs.cleaned_text == winning_plate), 4
            )

            # Evaluate Stability
            if (
                self.valid_observation_count >= min_valid_observations
                and self.plate_confidence >= min_stability_score
            ):
                self.plate_status = "stable"
            else:
                self.plate_status = "tentative"
        else:
            # Fallback when no observation strictly conformed to baseline regex
            if self.observations:
                best_obs = max(self.observations, key=lambda o: o.ocr_confidence)
                if best_obs.cleaned_text and best_obs.ocr_confidence >= 0.50:
                    self.plate_text = best_obs.cleaned_text
                    self.plate_confidence = round(best_obs.ocr_confidence, 4)
                    self.best_ocr_confidence = round(best_obs.ocr_confidence, 4)
                    self.plate_status = "tentative"
                else:
                    self.plate_text = ""
                    self.plate_confidence = 0.0
                    self.best_ocr_confidence = 0.0
                    self.plate_status = "unknown"
            else:
                self.plate_text = ""
                self.plate_confidence = 0.0
                self.best_ocr_confidence = 0.0
                self.plate_status = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes track record to standard JSON-compatible dictionary."""
        return {
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "plate_text": self.plate_text,
            "plate_status": self.plate_status,
            "plate_confidence": self.plate_confidence,
            "observation_count": self.observation_count,
            "valid_observation_count": self.valid_observation_count,
            "best_ocr_confidence": self.best_ocr_confidence,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
        }


class PlateTrackPipeline:
    """
    Complete Video Perception Pipeline:
    - Vehicle Detection & Tracking (YOLOv8 + ByteTrack)
    - License Plate Detection (YOLOv8)
    - Geometric Plate ↔ Vehicle Association
    - OCR Sampling & Temporal Aggregation
    - Annotated Visualization Video Generation
    """

    def __init__(
        self,
        vehicle_model_path: str = DEFAULT_VEHICLE_MODEL,
        plate_model_path: str = DEFAULT_PLATE_MODEL,
        tracker_config_path: str = DEFAULT_TRACKER_CONFIG,
        vehicle_conf: float = 0.40,
        plate_conf: float = 0.40,
        ocr_conf: float = 0.50,
        ocr_interval: int = 5,
        min_valid_observations: int = 3,
        min_stability_score: float = 0.70,
        imgsz_vehicle: int = 640,
        imgsz_plate: int = 1280,
        device: Optional[str] = None,
        output_dir: str = "runs/pipeline",
    ):
        self.vehicle_model_path = vehicle_model_path
        self.plate_model_path = plate_model_path
        self.tracker_config_path = tracker_config_path
        self.vehicle_conf = float(vehicle_conf)
        self.plate_conf = float(plate_conf)
        self.ocr_conf = float(ocr_conf)
        self.ocr_interval = int(ocr_interval)
        self.min_valid_observations = int(min_valid_observations)
        self.min_stability_score = float(min_stability_score)
        self.imgsz_vehicle = int(imgsz_vehicle)
        self.imgsz_plate = int(imgsz_plate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Device selection (RTX 3050 GPU for YOLO models, CPU for OCR)
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"YOLO Execution Device: {self.device}")

        # 2. Load Model 1 (Vehicle Detector)
        logger.info(f"Loading Vehicle Detector from: {self.vehicle_model_path}")
        self.vehicle_model = YOLO(self.vehicle_model_path)
        self.vehicle_classes = self.vehicle_model.names

        # 3. Load Model 2 (License Plate Detector)
        logger.info(f"Loading Plate Detector from: {self.plate_model_path}")
        self.plate_model = YOLO(self.plate_model_path)

        # 4. Resolve ByteTrack Config
        tracker_file = Path(self.tracker_config_path).resolve()
        if tracker_file.exists():
            self.tracker_arg = str(tracker_file)
        else:
            self.tracker_arg = "bytetrack.yaml"
        logger.info(f"ByteTrack Tracker Config: {self.tracker_arg}")

        # 5. Initialize Model 3 (PaddleOCR on CPU)
        logger.info("Initializing LicensePlateOCR engine on CPU...")
        self.ocr = LicensePlateOCR(
            confidence_threshold=self.ocr_conf,
            enable_clahe=True,
            enable_fallback=True,
            use_gpu=False,
            lang="en",
        )

        # 6. Active Track Registry
        self.tracks: Dict[int, VehicleTrackState] = {}
        self.track_trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

        # 7. Pipeline Telemetry Counters
        self.metrics = {
            "total_frames": 0,
            "processed_frames": 0,
            "plate_detections": 0,
            "plate_associations": 0,
            "ocr_attempts": 0,
            "ocr_successful": 0,
            "detection_time_total": 0.0,
            "ocr_time_total": 0.0,
            "pipeline_time_total": 0.0,
        }

    @staticmethod
    def associate_plates_to_vehicles(
        plates: List[Dict[str, Any]],
        vehicles: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Optional[int]]]:
        """
        Geometrically associates detected license plates with active vehicle tracks.

        Method:
        1. Calculates the geometric center (cx, cy) of the plate bounding box.
        2. Determines which active vehicle bounding boxes contain (cx, cy).
        3. If exactly one vehicle contains the center:
             Associate plate with that vehicle's track ID.
        4. If multiple vehicles contain the center:
             Choose the vehicle having the highest IoU with the plate bounding box.
        5. If no vehicle contains the center:
             associated_track_id = None.
        """
        associations: List[Tuple[Dict[str, Any], Optional[int]]] = []

        for plate in plates:
            px1, py1, px2, py2 = plate["bbox"]
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0

            # Find vehicles whose bounding box contains the plate center
            candidate_vehicles = []
            for veh in vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    candidate_vehicles.append(veh)

            if len(candidate_vehicles) == 1:
                associated_id = candidate_vehicles[0]["track_id"]
            elif len(candidate_vehicles) > 1:
                # Disambiguate overlapping vehicle boxes using maximum IoU
                best_veh = max(
                    candidate_vehicles,
                    key=lambda v: compute_box_iou(plate["bbox"], v["bbox"])
                )
                associated_id = best_veh["track_id"]
            else:
                associated_id = None

            associations.append((plate, associated_id))

        return associations

    def draw_annotations(
        self,
        frame: np.ndarray,
        active_vehicles: List[Dict[str, Any]],
        current_plates: List[Tuple[Dict[str, Any], Optional[int]]],
        frame_idx: int,
        total_frames: int,
        fps: float,
    ) -> np.ndarray:
        """
        Renders clear, professional annotations on the output video frame:
        - Vehicle bounding boxes and Track ID banners
        - License plate status and confidence badges
        - Detected plate bounding boxes
        - Motion centroid trajectory trails
        - Top-left diagnostics Heads-Up Display (HUD)
        """
        annotated = frame.copy()
        h_img, w_img = annotated.shape[:2]

        # 1. Motion Trajectory Trails
        for track_id, trail in self.track_trails.items():
            if len(trail) < 2:
                continue
            pts = list(trail)
            for i in range(1, len(pts)):
                thickness = int(max(1, (i / len(pts)) * 3))
                cv2.line(annotated, pts[i - 1], pts[i], (0, 220, 255), thickness, cv2.LINE_AA)

        # 2. Vehicle Bounding Boxes & Tracking Badges
        for veh in active_vehicles:
            tid = veh["track_id"]
            x1, y1, x2, y2 = veh["bbox"]
            cname = veh["class_name"]

            track_state = self.tracks.get(tid)
            plate_status = track_state.plate_status if track_state else "unknown"
            plate_text = track_state.plate_text if track_state else ""
            plate_conf = track_state.plate_confidence if track_state else 0.0

            # Color styling based on plate stability state
            if plate_status == "stable":
                box_color = (0, 220, 0)       # Vibrant Green: Stable Verified Plate
                status_symbol = "[STABLE]"
            elif plate_status == "tentative":
                box_color = (0, 165, 255)     # Orange: Tentative Plate
                status_symbol = "[TENTATIVE]"
            else:
                box_color = CLASS_COLORS.get(veh["class_id"], DEFAULT_VEHICLE_COLOR)
                status_symbol = ""

            # Draw vehicle boundary box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Build vehicle banner: Line 1 = Track ID & Class
            line1 = f"{cname.upper()} | ID:{tid}"
            # Line 2 = Plate & Confidence
            if plate_status == "stable":
                line2 = f"PLATE: {plate_text} ({plate_conf:.2f})"
            elif plate_status == "tentative":
                line2 = f"PLATE: {plate_text}? ({plate_conf:.2f})"
            else:
                line2 = "PLATE: UNKNOWN"

            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.52
            f_thick = 1
            (w1, h1), _ = cv2.getTextSize(line1, font, f_scale, f_thick)
            (w2, h2), _ = cv2.getTextSize(line2, font, f_scale, f_thick)

            badge_w = max(w1, w2) + 12
            badge_h = h1 + h2 + 14

            badge_y1 = max(0, y1 - badge_h - 4)
            badge_y2 = badge_y1 + badge_h
            badge_x2 = min(w_img, x1 + badge_w)

            # Draw solid badge background
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)

            # Text color (dark on bright badge, white on dark)
            text_color = (0, 0, 0) if (box_color[0]*0.299 + box_color[1]*0.587 + box_color[2]*0.114) > 130 else (255, 255, 255)

            cv2.putText(
                annotated,
                line1,
                (x1 + 6, badge_y1 + h1 + 4),
                font,
                f_scale,
                text_color,
                f_thick,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                line2,
                (x1 + 6, badge_y2 - 5),
                font,
                f_scale,
                text_color,
                f_thick,
                cv2.LINE_AA,
            )

        # 3. Plate Bounding Boxes
        for plate, assoc_id in current_plates:
            px1, py1, px2, py2 = plate["bbox"]
            p_conf = plate["conf"]
            p_color = (0, 255, 255) if assoc_id is not None else (0, 0, 255)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), p_color, 2, cv2.LINE_AA)

        # 4. Heads-Up Display (HUD) Dashboard (Top-Left)
        hud_h = 44
        hud_w = min(w_img, 680)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)

        # Aggregate current pipeline counts
        stable_count = sum(1 for t in self.tracks.values() if t.plate_status == "stable")
        tentative_count = sum(1 for t in self.tracks.values() if t.plate_status == "tentative")
        total_tracks = len(self.tracks)

        prog_str = f"Frame {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame {frame_idx}"
        hud_line1 = f"NETRA AI Engine | {prog_str} | {fps:.1f} FPS"
        hud_line2 = f"Tracks: {total_tracks} | Stable Plates: {stable_count} | Tentative: {tentative_count}"

        cv2.putText(annotated, hud_line1, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, hud_line2, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def process_video(
        self,
        video_source: str,
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end video tracking and temporal OCR pipeline on a video file.
        """
        source_path = Path(video_source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input video file not found: {source_path}")

        # Set default outputs
        if output_video_path is None:
            out_video = self.output_dir / "traffic_plate_tracking.mp4"
        else:
            out_video = Path(output_video_path).resolve()
        out_video.parent.mkdir(parents=True, exist_ok=True)

        if output_json_path is None:
            out_json = self.output_dir / "traffic_plate_tracking.json"
        else:
            out_json = Path(output_json_path).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video source: {source_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0.0 or video_fps > 120.0:
            video_fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video Source: {source_path.name} ({width}x{height} @ {video_fps:.2f} FPS, {total_frames} frames)")
        logger.info(f"Output Video: {out_video}")
        logger.info(f"Output Telemetry: {out_json}")

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, video_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open VideoWriter at: {out_video}")

        frame_idx = 0
        pipeline_start_time = time.perf_counter()
        rolling_fps = video_fps

        try:
            while True:
                t_frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                # -------------------------------------------------------------
                # 1. Vehicle Detection & ByteTrack
                # -------------------------------------------------------------
                t_det_start = time.perf_counter()
                track_results = self.vehicle_model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_arg,
                    conf=self.vehicle_conf,
                    imgsz=self.imgsz_vehicle,
                    device=self.device,
                    verbose=False,
                )

                # -------------------------------------------------------------
                # 2. License Plate Detection
                # -------------------------------------------------------------
                plate_results = self.plate_model.predict(
                    source=frame,
                    conf=self.plate_conf,
                    imgsz=self.imgsz_plate,
                    device=self.device,
                    verbose=False,
                )
                t_det_end = time.perf_counter()
                self.metrics["detection_time_total"] += (t_det_end - t_det_start)

                # Parse tracked vehicles
                active_vehicles: List[Dict[str, Any]] = []
                boxes_veh = track_results[0].boxes if len(track_results) > 0 else None

                if boxes_veh is not None and len(boxes_veh) > 0:
                    xyxy_v = boxes_veh.xyxy.cpu().numpy()
                    classes_v = boxes_veh.cls.cpu().numpy().astype(int)
                    confs_v = boxes_veh.conf.cpu().numpy()
                    has_ids = boxes_veh.id is not None
                    track_ids_v = boxes_veh.id.int().cpu().tolist() if has_ids else [-1] * len(boxes_veh)

                    for b_idx in range(len(boxes_veh)):
                        tid = int(track_ids_v[b_idx])
                        if tid <= 0:
                            continue  # Skip unassigned tracks

                        cid = int(classes_v[b_idx])
                        conf_val = float(confs_v[b_idx])
                        vx1, vy1, vx2, vy2 = map(int, xyxy_v[b_idx])
                        cname = self.vehicle_classes.get(cid, str(cid))

                        # Register or update track in memory
                        if tid not in self.tracks:
                            self.tracks[tid] = VehicleTrackState(
                                track_id=tid,
                                vehicle_class=cname,
                                first_seen_frame=frame_idx,
                            )

                        v_state = self.tracks[tid]
                        v_state.last_seen_frame = frame_idx
                        v_state.last_bbox = (vx1, vy1, vx2, vy2)

                        # Update trajectory trail
                        cx, cy = int((vx1 + vx2) / 2.0), int((vy1 + vy2) / 2.0)
                        self.track_trails[tid].append((cx, cy))

                        active_vehicles.append({
                            "track_id": tid,
                            "class_id": cid,
                            "class_name": cname,
                            "bbox": (vx1, vy1, vx2, vy2),
                            "conf": conf_val,
                        })

                # Parse detected license plates
                detected_plates: List[Dict[str, Any]] = []
                boxes_p = plate_results[0].boxes if len(plate_results) > 0 else None

                if boxes_p is not None and len(boxes_p) > 0:
                    xyxy_p = boxes_p.xyxy.cpu().numpy()
                    confs_p = boxes_p.conf.cpu().numpy()

                    for p_idx in range(len(boxes_p)):
                        px1, py1, px2, py2 = map(int, xyxy_p[p_idx])
                        # Clamp safely to frame boundary
                        px1 = max(0, min(width - 1, px1))
                        py1 = max(0, min(height - 1, py1))
                        px2 = max(px1 + 1, min(width, px2))
                        py2 = max(py1 + 1, min(height, py2))

                        pconf = float(confs_p[p_idx])
                        detected_plates.append({
                            "bbox": (px1, py1, px2, py2),
                            "conf": pconf,
                        })
                        self.metrics["plate_detections"] += 1

                # -------------------------------------------------------------
                # 3. Geometric Association (Plate ↔ Vehicle Track ID)
                # -------------------------------------------------------------
                associated_pairs = self.associate_plates_to_vehicles(
                    detected_plates, active_vehicles
                )

                # -------------------------------------------------------------
                # 4. OCR Execution with Sampling Interval
                # -------------------------------------------------------------
                t_ocr_frame_start = time.perf_counter()

                for plate, assoc_track_id in associated_pairs:
                    if assoc_track_id is None:
                        continue  # Plate not associated with an active vehicle track

                    self.metrics["plate_associations"] += 1
                    track_state = self.tracks[assoc_track_id]
                    track_state.last_plate_bbox = plate["bbox"]

                    # Apply OCR Sampling Interval (Throttle compute on CPU)
                    frames_since_last = frame_idx - track_state.last_ocr_frame
                    if frames_since_last >= self.ocr_interval or track_state.last_ocr_frame < 0:
                        # Extract plate crop
                        px1, py1, px2, py2 = plate["bbox"]
                        crop = frame[py1:py2, px1:px2]

                        if crop.size > 0:
                            self.metrics["ocr_attempts"] += 1
                            track_state.last_ocr_frame = frame_idx

                            # Execute LicensePlateOCR
                            ocr_res = self.ocr.predict(crop)
                            raw_txt = ocr_res.get("raw_text", "")
                            clean_txt = ocr_res.get("text", "")
                            ocr_c = float(ocr_res.get("confidence", 0.0))
                            valid_fmt = bool(ocr_res.get("valid_format", False))

                            if clean_txt:
                                self.metrics["ocr_successful"] += 1

                            obs = OCRObservation(
                                frame_idx=frame_idx,
                                raw_text=raw_txt,
                                cleaned_text=clean_txt,
                                ocr_confidence=ocr_c,
                                valid_format=valid_fmt,
                            )
                            track_state.add_observation(
                                obs,
                                min_valid_observations=self.min_valid_observations,
                                min_stability_score=self.min_stability_score,
                            )

                t_ocr_frame_end = time.perf_counter()
                self.metrics["ocr_time_total"] += (t_ocr_frame_end - t_ocr_frame_start)

                # -------------------------------------------------------------
                # 5. Render Annotations & Write Frame
                # -------------------------------------------------------------
                dt_frame = time.perf_counter() - t_frame_start
                current_fps = 1.0 / dt_frame if dt_frame > 0 else video_fps
                rolling_fps = 0.90 * rolling_fps + 0.10 * current_fps

                annotated_frame = self.draw_annotations(
                    frame=frame,
                    active_vehicles=active_vehicles,
                    current_plates=associated_pairs,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=rolling_fps,
                )

                writer.write(annotated_frame)

                if frame_idx % 25 == 0 or frame_idx == total_frames:
                    st_count = sum(1 for t in self.tracks.values() if t.plate_status == "stable")
                    ten_count = sum(1 for t in self.tracks.values() if t.plate_status == "tentative")
                    logger.info(
                        f"Frame {frame_idx:03d}/{total_frames} | "
                        f"FPS: {rolling_fps:.1f} | "
                        f"Active: {len(active_vehicles)} | "
                        f"Total Tracks: {len(self.tracks)} | "
                        f"Stable: {st_count} | Tentative: {ten_count}"
                    )

        finally:
            cap.release()
            writer.release()

        total_elapsed = time.perf_counter() - pipeline_start_time
        self.metrics["total_frames"] = total_frames
        self.metrics["processed_frames"] = frame_idx
        self.metrics["pipeline_time_total"] = total_elapsed
        pipeline_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0

        # Compile final track records
        final_track_records = [
            track.to_dict() for track in sorted(self.tracks.values(), key=lambda t: t.track_id)
        ]

        # Calculate stability metrics
        stable_tracks = [t for t in final_track_records if t["plate_status"] == "stable"]
        tentative_tracks = [t for t in final_track_records if t["plate_status"] == "tentative"]
        unknown_tracks = [t for t in final_track_records if t["plate_status"] == "unknown"]

        avg_ocr_ms = (
            (self.metrics["ocr_time_total"] / self.metrics["ocr_attempts"] * 1000.0)
            if self.metrics["ocr_attempts"] > 0
            else 0.0
        )

        telemetry_payload = {
            "metadata": {
                "source_video": str(source_path),
                "resolution": [width, height],
                "video_fps": video_fps,
                "total_video_frames": total_frames,
                "processed_frames": frame_idx,
                "pipeline_processing_fps": round(pipeline_fps, 2),
                "total_processing_time_sec": round(total_elapsed, 2),
                "output_video_path": str(out_video),
            },
            "parameters": {
                "vehicle_detector_weights": self.vehicle_model_path,
                "plate_detector_weights": self.plate_model_path,
                "vehicle_conf_threshold": self.vehicle_conf,
                "plate_conf_threshold": self.plate_conf,
                "ocr_conf_threshold": self.ocr_conf,
                "ocr_interval_frames": self.ocr_interval,
                "min_valid_observations": self.min_valid_observations,
                "min_stability_score": self.min_stability_score,
            },
            "summary_statistics": {
                "total_vehicle_tracks": len(final_track_records),
                "total_plate_detections": self.metrics["plate_detections"],
                "total_plate_associations": self.metrics["plate_associations"],
                "total_ocr_attempts": self.metrics["ocr_attempts"],
                "successful_ocr_results": self.metrics["ocr_successful"],
                "stable_plate_tracks": len(stable_tracks),
                "tentative_plate_tracks": len(tentative_tracks),
                "unknown_plate_tracks": len(unknown_tracks),
                "average_ocr_latency_ms": round(avg_ocr_ms, 2),
            },
            "vehicle_tracks": final_track_records,
        }

        # Save telemetry JSON
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(telemetry_payload, jf, indent=2)

        logger.info(f"Video processing finished. Telemetry written to: {out_json}")
        return telemetry_payload


def main():
    """Command-line entry point for running the video tracking & OCR pipeline."""
    parser = argparse.ArgumentParser(
        description="NETRA Video Perception Pipeline: ByteTrack + Plate Detection + License Plate OCR"
    )
    parser.add_argument("--source", type=str, default="data/videos/test_2.mp4", help="Path to input video file")
    parser.add_argument("--vehicle-model", type=str, default=DEFAULT_VEHICLE_MODEL, help="Path to vehicle YOLO best.pt")
    parser.add_argument("--plate-model", type=str, default=DEFAULT_PLATE_MODEL, help="Path to plate YOLO best.pt")
    parser.add_argument("--tracker-config", type=str, default=DEFAULT_TRACKER_CONFIG, help="Path to tracker YAML")
    parser.add_argument("--vehicle-conf", type=float, default=0.40, help="Vehicle detection confidence threshold")
    parser.add_argument("--plate-conf", type=float, default=0.40, help="Plate detection confidence threshold")
    parser.add_argument("--ocr-conf", type=float, default=0.50, help="OCR confidence threshold")
    parser.add_argument("--ocr-interval", type=int, default=5, help="OCR sampling interval in frames per track")
    parser.add_argument("--min-valid-obs", type=int, default=3, help="Minimum valid observations for stable plate")
    parser.add_argument("--min-stability", type=float, default=0.70, help="Minimum stability score for stable plate")
    parser.add_argument("--output-dir", type=str, default="runs/pipeline", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames to process for testing")

    args = parser.parse_args()

    pipeline = PlateTrackPipeline(
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        tracker_config_path=args.tracker_config,
        vehicle_conf=args.vehicle_conf,
        plate_conf=args.plate_conf,
        ocr_conf=args.ocr_conf,
        ocr_interval=args.ocr_interval,
        min_valid_observations=args.min_valid_obs,
        min_stability_score=args.min_stability,
        output_dir=args.output_dir,
    )

    results = pipeline.process_video(
        video_source=args.source,
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 70)
    print("NETRA VIDEO PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    stats = results["summary_statistics"]
    meta = results["metadata"]
    print(f"Processed Frames:       {meta['processed_frames']} / {meta['total_video_frames']}")
    print(f"Processing Speed:       {meta['pipeline_processing_fps']:.1f} FPS (Total: {meta['total_processing_time_sec']:.1f}s)")
    print(f"Total Vehicle Tracks:   {stats['total_vehicle_tracks']}")
    print(f"Plate Detections:       {stats['total_plate_detections']}")
    print(f"Plate Associations:     {stats['total_plate_associations']}")
    print(f"OCR Attempts:           {stats['total_ocr_attempts']}")
    print(f"Successful OCR Reads:   {stats['successful_ocr_results']}")
    print(f"Stable Plates:          {stats['stable_plate_tracks']}")
    print(f"Tentative Plates:       {stats['tentative_plate_tracks']}")
    print(f"Unknown Tracks:         {stats['unknown_plate_tracks']}")
    print(f"Average OCR Latency:    {stats['average_ocr_latency_ms']:.1f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
