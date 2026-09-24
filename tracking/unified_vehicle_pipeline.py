"""
NETRA — Checkpoint 5: Unified Vehicle Track Record Pipeline
Module: tracking/unified_vehicle_pipeline.py
Project: SIH 2026 NETRA (Networked Engine for Traffic Recognition & Analytics)

Pipeline Orchestration:
    Video Stream
         ↓
    Vehicle Detection (YOLOv8n UVH-26) ──> GPU
         ↓
    ByteTrack (Kalman Motion Filter + 2-Stage IoU)
         ↓
    Local Vehicle Track ID & Bounding Box
         ├──> Vehicle Crop ──> OSNet-AIN Re-ID ──> 512-D Normalized Embedding (GPU)
         │                       └─> Temporal Average & L2 Normalization (Representative Embedding)
         └──> License Plate Detection (YOLOv8n Plate Detector, imgsz=1280) ──> GPU
                   ↓
              Geometric Association (Plate Center Containment / IoU)
                   ↓
              OCR Sampling Check (Throttle: Frame Δ >= 5 per track)
                   ↓
              PaddleOCR (det=False, rec=True, cls=True) ──> CPU
                   ↓
              Temporal Confidence-Weighted Voting & Stability Evaluation
                   ↓
    Unified VehicleTrackRecord (ByteTrack ID + OSNet Re-ID + License Plate OCR)
         ↓
    Annotated Video (runs/pipeline/unified_vehicle_tracks.mp4)
    Structured JSON (runs/pipeline/unified_vehicle_tracks.json)

IMPORTANT IDENTITY DISTINCTION:
    - ByteTrack Track ID: Strictly a LOCAL identity valid ONLY within this single video/camera feed.
    - OSNet-AIN Re-ID: A 512-dimensional visual appearance representation for vehicle similarity.
    - Plate OCR: Physical registration evidence extracted from license plate crops.
    - Global Vehicle ID: Cross-camera global identity matching is NOT implemented in this checkpoint.
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
from reid.vehicle_reid import VehicleReIDExtractor

# Configure module logger
logger = logging.getLogger("NETRA.UnifiedPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [NETRA.Unified] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default Model & Weight Paths
DEFAULT_VEHICLE_MODEL = "runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/best.pt"
DEFAULT_PLATE_MODEL = "runs/detect/runs/plate_detection/vehicle_plate_yolov8n_50ep/weights/best.pt"
DEFAULT_REID_WEIGHTS = "weights/osnet_ain_x1_0_vehicle_reid.pt"
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


class UnifiedVehicleTrack:
    """
    Maintains complete state for a single vehicle track:
    - ByteTrack local identity & temporal lifecycle
    - OSNet-AIN appearance embeddings & representative vector
    - License plate observations & temporal confidence-weighted voting
    """

    def __init__(self, track_id: int, vehicle_class: str, first_seen_frame: int):
        self.track_id: int = int(track_id)
        self.vehicle_class: str = str(vehicle_class)
        self.first_seen_frame: int = int(first_seen_frame)
        self.last_seen_frame: int = int(first_seen_frame)
        self.last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.last_plate_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_detector_conf: float = 0.0

        # Sampling throttles
        self.last_ocr_frame: int = -999
        self.last_reid_frame: int = -999

        # Re-ID Embedding History
        self.reid_embeddings: List[np.ndarray] = []
        self.representative_embedding: Optional[np.ndarray] = None

        # License Plate OCR Observation History
        self.plate_observations: List[OCRObservation] = []
        self.plate_text: Optional[str] = None
        self.plate_status: str = "unknown"  # "unknown" | "tentative" | "stable"
        self.plate_confidence: float = 0.0
        self.best_ocr_confidence: float = 0.0
        self.plate_observation_count: int = 0
        self.valid_plate_observation_count: int = 0

    @property
    def frame_count(self) -> int:
        """Total frame duration over which the vehicle was visible."""
        return max(1, self.last_seen_frame - self.first_seen_frame + 1)

    # -------------------------------------------------------------------------
    # Re-ID Embedding Management
    # -------------------------------------------------------------------------

    def add_reid_embedding(self, embedding: np.ndarray) -> bool:
        """
        Appends a valid 512-D L2-normalized embedding and updates the representative vector.
        """
        if not isinstance(embedding, np.ndarray):
            return False
        if embedding.shape != (512,):
            return False
        if not np.all(np.isfinite(embedding)):
            return False

        norm = float(np.linalg.norm(embedding))
        if abs(norm - 1.0) > 0.08:
            # Re-normalize if slight numerical drift
            if norm > 1e-12:
                embedding = (embedding / norm).astype(np.float32)
            else:
                return False

        self.reid_embeddings.append(embedding)

        # Update representative embedding via mean vector + L2-normalization
        mean_vec = np.mean(self.reid_embeddings, axis=0)
        mean_norm = float(np.linalg.norm(mean_vec))
        if mean_norm > 1e-12:
            self.representative_embedding = (mean_vec / mean_norm).astype(np.float32)
        else:
            self.representative_embedding = mean_vec.astype(np.float32)

        return True

    # -------------------------------------------------------------------------
    # License Plate Temporal Voting Management
    # -------------------------------------------------------------------------

    def add_plate_observation(
        self,
        obs: OCRObservation,
        min_valid_observations: int = 3,
        min_stability_score: float = 0.70,
    ) -> None:
        """Adds an OCR observation and updates temporal voting state."""
        self.plate_observations.append(obs)
        self.update_plate_aggregation(
            min_valid_observations=min_valid_observations,
            min_stability_score=min_stability_score,
        )

    def update_plate_aggregation(
        self,
        min_valid_observations: int = 3,
        min_stability_score: float = 0.70,
    ) -> None:
        """
        Executes confidence-weighted voting across recorded plate observations:
        - Only observations with valid_format=True and confidence >= 0.50 cast primary votes.
        - Votes are weighted by OCR model confidence.
        - Stability rule: >= MIN_VALID_OBSERVATIONS and plate_confidence >= MIN_STABILITY_SCORE.
        """
        self.plate_observation_count = len(self.plate_observations)
        valid_obs = [
            obs for obs in self.plate_observations
            if obs.valid_format and obs.ocr_confidence >= 0.50
        ]
        self.valid_plate_observation_count = len(valid_obs)

        if valid_obs:
            vote_weights: Dict[str, float] = defaultdict(float)
            vote_counts: Dict[str, int] = defaultdict(int)

            for obs in valid_obs:
                vote_weights[obs.cleaned_text] += obs.ocr_confidence
                vote_counts[obs.cleaned_text] += 1

            winning_plate = max(vote_weights.keys(), key=lambda k: vote_weights[k])
            self.plate_text = winning_plate
            self.plate_confidence = round(
                vote_weights[winning_plate] / float(vote_counts[winning_plate]), 4
            )
            self.best_ocr_confidence = round(
                max(obs.ocr_confidence for obs in valid_obs if obs.cleaned_text == winning_plate), 4
            )

            if (
                self.valid_plate_observation_count >= min_valid_observations
                and self.plate_confidence >= min_stability_score
            ):
                self.plate_status = "stable"
            else:
                self.plate_status = "tentative"
        else:
            if self.plate_observations:
                best_obs = max(self.plate_observations, key=lambda o: o.ocr_confidence)
                if best_obs.cleaned_text and best_obs.ocr_confidence >= 0.50:
                    self.plate_text = best_obs.cleaned_text
                    self.plate_confidence = round(best_obs.ocr_confidence, 4)
                    self.best_ocr_confidence = round(best_obs.ocr_confidence, 4)
                    self.plate_status = "tentative"
                else:
                    self.plate_text = None
                    self.plate_confidence = 0.0
                    self.best_ocr_confidence = 0.0
                    self.plate_status = "unknown"
            else:
                self.plate_text = None
                self.plate_confidence = 0.0
                self.best_ocr_confidence = 0.0
                self.plate_status = "unknown"

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_record_dict(self) -> Dict[str, Any]:
        """
        Builds the unified VehicleTrackRecord matching Checkpoint 5 schema.
        Includes 512-D representative embedding list and plate status attributes.
        """
        rep_list = (
            [round(float(x), 6) for x in self.representative_embedding]
            if self.representative_embedding is not None
            else None
        )

        # Validate last_bbox (x1 < x2 and y1 < y2 and not [0,0,0,0])
        valid_bbox = None
        if self.last_bbox and len(self.last_bbox) == 4:
            x1, y1, x2, y2 = map(int, self.last_bbox)
            if x1 < x2 and y1 < y2 and not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                valid_bbox = [x1, y1, x2, y2]

        det_conf = round(float(self.last_detector_conf), 4) if self.last_detector_conf > 0 else None
        best_ocr = round(float(self.best_ocr_confidence), 4) if self.best_ocr_confidence > 0 else None

        return {
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "frame_count": self.frame_count,
            "last_bbox": valid_bbox,
            "detector_confidence": det_conf,
            "reid": {
                "embedding_dimension": 512,
                "normalized": True,
                "observation_count": len(self.reid_embeddings),
                "representative_embedding": rep_list,
            },
            "plate": {
                "text": self.plate_text,
                "status": self.plate_status,
                "confidence": round(float(self.plate_confidence), 4),
                "observation_count": self.plate_observation_count,
                "valid_observation_count": self.valid_plate_observation_count,
                "best_ocr_confidence": best_ocr,
            },
        }


class UnifiedVehiclePipeline:
    """
    Unified Video Perception Pipeline combining:
    - Model 1: YOLOv8n UVH-26 Vehicle Detection + ByteTrack
    - OSNet-AIN: 512-D Vehicle Re-ID Feature Extraction & Aggregation
    - Model 2: YOLOv8n License Plate Detection
    - Geometric Association: Plate Center ↔ Vehicle Bounding Box Containment
    - Model 3: PaddleOCR Recognition-Only (CPU) with Temporal Voting
    - Telemetry & Visualization Video Generation
    """

    def __init__(
        self,
        vehicle_model_path: str = DEFAULT_VEHICLE_MODEL,
        plate_model_path: str = DEFAULT_PLATE_MODEL,
        reid_weights_path: Optional[str] = DEFAULT_REID_WEIGHTS,
        tracker_config_path: str = DEFAULT_TRACKER_CONFIG,
        vehicle_conf: float = 0.40,
        plate_conf: float = 0.35,
        ocr_conf: float = 0.50,
        ocr_interval: int = 5,
        reid_interval: int = 10,
        min_valid_observations: int = 3,
        min_stability_score: float = 0.70,
        imgsz_vehicle: int = 640,
        imgsz_plate: int = 1280,
        device: Optional[str] = None,
        output_dir: str = "runs/pipeline",
    ):
        self.vehicle_model_path = vehicle_model_path
        self.plate_model_path = plate_model_path
        self.reid_weights_path = reid_weights_path
        self.tracker_config_path = tracker_config_path
        self.vehicle_conf = float(vehicle_conf)
        self.plate_conf = float(plate_conf)
        self.ocr_conf = float(ocr_conf)
        self.ocr_interval = int(ocr_interval)
        self.reid_interval = int(reid_interval)
        self.min_valid_observations = int(min_valid_observations)
        self.min_stability_score = float(min_stability_score)
        self.imgsz_vehicle = int(imgsz_vehicle)
        self.imgsz_plate = int(imgsz_plate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Device Resolution (GPU for YOLO & Re-ID, CPU for OCR)
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"GPU Compute Device: {self.device}")

        # 2. Load Model 1 (Vehicle Detector)
        logger.info(f"Loading Vehicle Detector from: {self.vehicle_model_path}")
        self.vehicle_model = YOLO(self.vehicle_model_path)
        self.vehicle_classes = self.vehicle_model.names

        # 3. Load Model 2 (Plate Detector)
        logger.info(f"Loading Plate Detector from: {self.plate_model_path}")
        self.plate_model = YOLO(self.plate_model_path)

        # 4. Resolve ByteTrack Config
        tracker_file = Path(self.tracker_config_path).resolve()
        if tracker_file.exists():
            self.tracker_arg = str(tracker_file)
        else:
            self.tracker_arg = "bytetrack.yaml"
        logger.info(f"ByteTrack Tracker Config: {self.tracker_arg}")

        # 5. Initialize OSNet-AIN Vehicle Re-ID Extractor (GPU)
        logger.info("Initializing OSNet-AIN Vehicle Re-ID Extractor...")
        self.reid_extractor = VehicleReIDExtractor(
            weights_path=self.reid_weights_path,
            device=self.device,
        )
        logger.info("Vehicle Re-ID Extractor initialized successfully.")

        # 6. Initialize Model 3 (PaddleOCR on CPU)
        logger.info("Initializing LicensePlateOCR engine on CPU...")
        self.ocr = LicensePlateOCR(
            confidence_threshold=self.ocr_conf,
            enable_clahe=True,
            enable_fallback=True,
            use_gpu=False,
            lang="en",
        )
        logger.info("LicensePlateOCR engine initialized successfully.")

        # 7. Unified Track Registry & Visual Trails
        self.tracks: Dict[int, UnifiedVehicleTrack] = {}
        self.track_trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

        # 8. Pipeline Telemetry Metrics
        self.metrics = {
            "total_frames": 0,
            "processed_frames": 0,
            "plate_detections": 0,
            "plate_associations": 0,
            "ocr_attempts": 0,
            "ocr_successful": 0,
            "reid_attempts": 0,
            "reid_successful": 0,
            "detection_time_total": 0.0,
            "reid_time_total": 0.0,
            "ocr_time_total": 0.0,
            "pipeline_time_total": 0.0,
        }

    @staticmethod
    def associate_plates_to_vehicles(
        plates: List[Dict[str, Any]],
        vehicles: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Optional[int]]]:
        """
        Geometrically associates detected license plates with active vehicle tracks:
        1. Plate center (cx, cy) is tested for containment within vehicle bounding boxes.
        2. Exactly 1 vehicle containing center -> Direct association.
        3. Multiple vehicles containing center -> Maximum IoU disambiguation.
        4. Zero vehicles containing center -> Unassociated (None).
        """
        associations: List[Tuple[Dict[str, Any], Optional[int]]] = []

        for plate in plates:
            px1, py1, px2, py2 = plate["bbox"]
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0

            candidate_vehicles = []
            for veh in vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    candidate_vehicles.append(veh)

            if len(candidate_vehicles) == 1:
                associated_id = candidate_vehicles[0]["track_id"]
            elif len(candidate_vehicles) > 1:
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
        Renders standardized multi-modal annotations:
        - Vehicle bounding boxes and badges
        - Re-ID extraction status
        - Plate recognition & stability state
        - Plate bounding boxes
        - Motion centroid trajectory trails
        - Diagnostics Heads-Up Display (HUD)
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

        # 2. Vehicle Bounding Boxes & Multi-Modal Badges
        for veh in active_vehicles:
            tid = veh["track_id"]
            x1, y1, x2, y2 = veh["bbox"]
            cname = veh["class_name"]
            det_c = veh["conf"]

            track_state = self.tracks.get(tid)
            plate_status = track_state.plate_status if track_state else "unknown"
            plate_text = track_state.plate_text if track_state else None
            plate_conf = track_state.plate_confidence if track_state else 0.0
            reid_count = len(track_state.reid_embeddings) if track_state else 0

            # Color styling based on plate stability state
            if plate_status == "stable":
                box_color = (0, 220, 0)       # Vibrant Green: Verified Stable Plate
            elif plate_status == "tentative":
                box_color = (0, 165, 255)     # Orange: Tentative Plate
            else:
                box_color = CLASS_COLORS.get(veh["class_id"], DEFAULT_VEHICLE_COLOR)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Build Multi-Modal Banner
            # Line 1: Vehicle Class & Track ID
            line1 = f"{cname.upper()} | ID:{tid} | {det_c:.2f}"
            # Line 2: Re-ID Status
            line2 = f"REID: OK ({reid_count})" if reid_count > 0 else "REID: N/A"
            # Line 3: Plate Status
            if plate_status == "stable":
                line3 = f"PLATE: {plate_text} ({plate_conf:.2f})"
            elif plate_status == "tentative":
                line3 = f"PLATE: {plate_text}? ({plate_conf:.2f})"
            else:
                line3 = "PLATE: UNKNOWN"

            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.50
            f_thick = 1
            (w1, h1), _ = cv2.getTextSize(line1, font, f_scale, f_thick)
            (w2, h2), _ = cv2.getTextSize(line2, font, f_scale, f_thick)
            (w3, h3), _ = cv2.getTextSize(line3, font, f_scale, f_thick)

            badge_w = max(w1, w2, w3) + 14
            badge_h = h1 + h2 + h3 + 20

            badge_y1 = max(0, y1 - badge_h - 4)
            badge_y2 = badge_y1 + badge_h
            badge_x2 = min(w_img, x1 + badge_w)

            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)

            text_color = (0, 0, 0) if (box_color[0]*0.299 + box_color[1]*0.587 + box_color[2]*0.114) > 130 else (255, 255, 255)

            cv2.putText(annotated, line1, (x1 + 6, badge_y1 + h1 + 4), font, f_scale, text_color, f_thick, cv2.LINE_AA)
            cv2.putText(annotated, line2, (x1 + 6, badge_y1 + h1 + h2 + 10), font, f_scale, text_color, f_thick, cv2.LINE_AA)
            cv2.putText(annotated, line3, (x1 + 6, badge_y2 - 6), font, f_scale, text_color, f_thick, cv2.LINE_AA)

        # 3. Plate Bounding Boxes
        for plate, assoc_id in current_plates:
            px1, py1, px2, py2 = plate["bbox"]
            p_color = (0, 255, 255) if assoc_id is not None else (0, 0, 255)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), p_color, 2, cv2.LINE_AA)

        # 4. Heads-Up Display (HUD) Dashboard
        hud_h = 44
        hud_w = min(w_img, 720)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)

        stable_count = sum(1 for t in self.tracks.values() if t.plate_status == "stable")
        tentative_count = sum(1 for t in self.tracks.values() if t.plate_status == "tentative")
        total_tracks = len(self.tracks)
        total_reid_obs = sum(len(t.reid_embeddings) for t in self.tracks.values())

        prog_str = f"Frame {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame {frame_idx}"
        hud_line1 = f"NETRA Unified Pipeline | {prog_str} | {fps:.1f} FPS"
        hud_line2 = f"Tracks: {total_tracks} | Re-ID: {total_reid_obs} | Stable Plates: {stable_count} | Tentative: {tentative_count}"

        cv2.putText(annotated, hud_line1, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, hud_line2, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def reset(self) -> None:
        """Resets track registry, motion trails, ByteTrack internal trackers, and metrics."""
        self.tracks.clear()
        self.track_trails.clear()
        if hasattr(self.vehicle_model, "predictor") and self.vehicle_model.predictor is not None:
            predictor = self.vehicle_model.predictor
            if hasattr(predictor, "trackers"):
                if predictor.trackers:
                    for trk in predictor.trackers:
                        if hasattr(trk, "reset"):
                            trk.reset()
                delattr(predictor, "trackers")
        self.metrics = {
            "total_frames": 0,
            "processed_frames": 0,
            "plate_detections": 0,
            "plate_associations": 0,
            "ocr_attempts": 0,
            "ocr_successful": 0,
            "reid_attempts": 0,
            "reid_successful": 0,
            "detection_time_total": 0.0,
            "reid_time_total": 0.0,
            "ocr_time_total": 0.0,
            "pipeline_time_total": 0.0,
        }

    def process_video(
        self,
        video_source: str,
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        save_video: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes the unified video tracking, Re-ID embedding, and license plate OCR pipeline.
        """
        source_path = Path(video_source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input video file not found: {source_path}")

        # Set default output paths
        if output_video_path is None:
            out_video = self.output_dir / "unified_vehicle_tracks.mp4"
        else:
            out_video = Path(output_video_path).resolve()
        out_video.parent.mkdir(parents=True, exist_ok=True)

        if output_json_path is None:
            out_json = self.output_dir / "unified_vehicle_tracks.json"
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
        if save_video:
            logger.info(f"Output Video: {out_video}")
        logger.info(f"Output JSON:  {out_json}")

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, video_fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to open VideoWriter at: {out_video}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"Seeked to start_frame: {start_frame}")

        frame_idx = start_frame
        pipeline_start_time = time.perf_counter()
        rolling_fps = video_fps

        try:
            while True:
                t_frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if max_frames and (frame_idx - start_frame) > max_frames:
                    break
                if end_frame is not None and frame_idx > end_frame:
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

                # Parse active tracked vehicles
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
                            continue

                        cid = int(classes_v[b_idx])
                        conf_val = float(confs_v[b_idx])
                        vx1, vy1, vx2, vy2 = map(int, xyxy_v[b_idx])
                        cname = self.vehicle_classes.get(cid, str(cid))

                        # Safe clamping
                        vx1 = max(0, min(width - 1, vx1))
                        vy1 = max(0, min(height - 1, vy1))
                        vx2 = max(vx1 + 1, min(width, vx2))
                        vy2 = max(vy1 + 1, min(height, vy2))

                        if tid not in self.tracks:
                            self.tracks[tid] = UnifiedVehicleTrack(
                                track_id=tid,
                                vehicle_class=cname,
                                first_seen_frame=frame_idx,
                            )

                        v_state = self.tracks[tid]
                        v_state.last_seen_frame = frame_idx
                        v_state.last_bbox = (vx1, vy1, vx2, vy2)
                        v_state.last_detector_conf = conf_val

                        # Update motion trail
                        cx, cy = int((vx1 + vx2) / 2.0), int((vy1 + vy2) / 2.0)
                        self.track_trails[tid].append((cx, cy))

                        active_vehicles.append({
                            "track_id": tid,
                            "class_id": cid,
                            "class_name": cname,
                            "bbox": (vx1, vy1, vx2, vy2),
                            "conf": conf_val,
                        })

                        # -----------------------------------------------------
                        # 3. OSNet-AIN Vehicle Re-ID Sampling
                        # -----------------------------------------------------
                        frames_since_reid = frame_idx - v_state.last_reid_frame
                        if frames_since_reid >= self.reid_interval or v_state.last_reid_frame < 0:
                            crop_w = vx2 - vx1
                            crop_h = vy2 - vy1
                            # Minimum crop dimension check
                            if crop_w >= 20 and crop_h >= 20:
                                v_crop = frame[vy1:vy2, vx1:vx2]
                                t_reid_start = time.perf_counter()
                                self.metrics["reid_attempts"] += 1
                                try:
                                    emb = self.reid_extractor.extract_embedding(v_crop)
                                    if v_state.add_reid_embedding(emb):
                                        self.metrics["reid_successful"] += 1
                                        v_state.last_reid_frame = frame_idx
                                except Exception as reid_err:
                                    logger.warning(f"Re-ID extraction failed on track {tid}: {reid_err}")
                                finally:
                                    self.metrics["reid_time_total"] += (time.perf_counter() - t_reid_start)

                # Parse detected license plates
                detected_plates: List[Dict[str, Any]] = []
                boxes_p = plate_results[0].boxes if len(plate_results) > 0 else None

                if boxes_p is not None and len(boxes_p) > 0:
                    xyxy_p = boxes_p.xyxy.cpu().numpy()
                    confs_p = boxes_p.conf.cpu().numpy()

                    for p_idx in range(len(boxes_p)):
                        px1, py1, px2, py2 = map(int, xyxy_p[p_idx])
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
                # 4. Geometric Plate ↔ Vehicle Association
                # -------------------------------------------------------------
                associated_pairs = self.associate_plates_to_vehicles(
                    detected_plates, active_vehicles
                )

                # -------------------------------------------------------------
                # 5. License Plate OCR Sampling & Temporal Voting
                # -------------------------------------------------------------
                t_ocr_frame_start = time.perf_counter()

                for plate, assoc_track_id in associated_pairs:
                    if assoc_track_id is None:
                        continue

                    self.metrics["plate_associations"] += 1
                    track_state = self.tracks[assoc_track_id]
                    track_state.last_plate_bbox = plate["bbox"]

                    frames_since_ocr = frame_idx - track_state.last_ocr_frame
                    if frames_since_ocr >= self.ocr_interval or track_state.last_ocr_frame < 0:
                        px1, py1, px2, py2 = plate["bbox"]
                        plate_crop = frame[py1:py2, px1:px2]

                        if plate_crop.size > 0:
                            self.metrics["ocr_attempts"] += 1
                            track_state.last_ocr_frame = frame_idx

                            try:
                                ocr_res = self.ocr.predict(plate_crop)
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
                                track_state.add_plate_observation(
                                    obs,
                                    min_valid_observations=self.min_valid_observations,
                                    min_stability_score=self.min_stability_score,
                                )
                            except Exception as ocr_err:
                                logger.warning(f"OCR inference failed on track {assoc_track_id}: {ocr_err}")

                self.metrics["ocr_time_total"] += (time.perf_counter() - t_ocr_frame_start)

                # -------------------------------------------------------------
                # 6. Render Annotations & Write Frame
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

                if writer is not None:
                    writer.write(annotated_frame)

                if frame_idx % 25 == 0 or frame_idx == total_frames:
                    st_count = sum(1 for t in self.tracks.values() if t.plate_status == "stable")
                    ten_count = sum(1 for t in self.tracks.values() if t.plate_status == "tentative")
                    total_reid = sum(len(t.reid_embeddings) for t in self.tracks.values())
                    logger.info(
                        f"Frame {frame_idx:03d}/{total_frames} | "
                        f"FPS: {rolling_fps:.1f} | "
                        f"Active: {len(active_vehicles)} | "
                        f"Tracks: {len(self.tracks)} | "
                        f"Re-ID: {total_reid} | "
                        f"Stable Plates: {st_count} | Tentative: {ten_count}"
                    )

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        total_elapsed = time.perf_counter() - pipeline_start_time
        self.metrics["total_frames"] = total_frames
        self.metrics["processed_frames"] = frame_idx
        self.metrics["pipeline_time_total"] = total_elapsed
        pipeline_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0

        # Compile final unified track records
        unified_records = [
            track.to_record_dict()
            for track in sorted(self.tracks.values(), key=lambda t: t.track_id)
        ]

        # Calculate summary statistics
        total_tracks_count = len(unified_records)
        tracks_with_reid = sum(1 for r in unified_records if r["reid"]["observation_count"] > 0)
        tracks_with_plate = sum(1 for r in unified_records if r["plate"]["observation_count"] > 0)
        stable_tracks = sum(1 for r in unified_records if r["plate"]["status"] == "stable")
        tentative_tracks = sum(1 for r in unified_records if r["plate"]["status"] == "tentative")
        unknown_tracks = sum(1 for r in unified_records if r["plate"]["status"] == "unknown")

        avg_ocr_ms = (
            (self.metrics["ocr_time_total"] / self.metrics["ocr_attempts"] * 1000.0)
            if self.metrics["ocr_attempts"] > 0
            else 0.0
        )
        avg_reid_ms = (
            (self.metrics["reid_time_total"] / self.metrics["reid_attempts"] * 1000.0)
            if self.metrics["reid_attempts"] > 0
            else 0.0
        )

        telemetry_payload = {
            "metadata": {
                "source": str(source_path),
                "total_frames": total_frames,
                "fps": round(video_fps, 2),
                "resolution": f"{width}x{height}",
                "reid_sampling_interval": self.reid_interval,
                "ocr_sampling_interval": self.ocr_interval,
                "processed_frames": frame_idx,
                "pipeline_fps": round(pipeline_fps, 2),
                "total_processing_time_sec": round(total_elapsed, 2),
                "output_video_path": str(out_video),
                "device": self.device,
                "note_on_identity": (
                    "Track IDs represent purely local single-camera ByteTrack associations. "
                    "Global multi-camera identity matching and trajectory reconstruction are intentionally "
                    "deferred to subsequent phases."
                ),
            },
            "parameters": {
                "vehicle_detector_weights": self.vehicle_model_path,
                "plate_detector_weights": self.plate_model_path,
                "reid_model_weights": str(self.reid_weights_path),
                "vehicle_conf_threshold": self.vehicle_conf,
                "plate_conf_threshold": self.plate_conf,
                "ocr_conf_threshold": self.ocr_conf,
                "min_valid_observations": self.min_valid_observations,
                "min_stability_score": self.min_stability_score,
            },
            "summary_statistics": {
                "total_vehicle_tracks": total_tracks_count,
                "tracks_with_reid": tracks_with_reid,
                "total_reid_observations": self.metrics["reid_successful"],
                "tracks_with_plate_observations": tracks_with_plate,
                "stable_plate_tracks": stable_tracks,
                "tentative_plate_tracks": tentative_tracks,
                "unknown_plate_tracks": unknown_tracks,
                "total_plate_detections": self.metrics["plate_detections"],
                "total_plate_associations": self.metrics["plate_associations"],
                "total_ocr_attempts": self.metrics["ocr_attempts"],
                "successful_ocr_reads": self.metrics["ocr_successful"],
                "average_ocr_latency_ms": round(avg_ocr_ms, 2),
                "average_reid_latency_ms": round(avg_reid_ms, 2),
            },
            "vehicle_tracks": unified_records,
        }

        # Write output JSON
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(telemetry_payload, jf, indent=2)

        logger.info(f"Unified tracking finished. Telemetry written to: {out_json}")
        return telemetry_payload


def main():
    """Command-line entrypoint for UnifiedVehiclePipeline."""
    parser = argparse.ArgumentParser(
        description="NETRA Unified Vehicle Pipeline: ByteTrack + OSNet Re-ID + License Plate OCR"
    )
    parser.add_argument("--source", type=str, default="data/videos/test_2.mp4", help="Path to input video")
    parser.add_argument("--vehicle-model", type=str, default=DEFAULT_VEHICLE_MODEL, help="Vehicle YOLO weights")
    parser.add_argument("--plate-model", type=str, default=DEFAULT_PLATE_MODEL, help="Plate YOLO weights")
    parser.add_argument("--reid-weights", type=str, default=DEFAULT_REID_WEIGHTS, help="OSNet-AIN Re-ID weights")
    parser.add_argument("--tracker-config", type=str, default=DEFAULT_TRACKER_CONFIG, help="Tracker YAML config")
    parser.add_argument("--vehicle-conf", type=float, default=0.40, help="Vehicle confidence threshold")
    parser.add_argument("--plate-conf", type=float, default=0.35, help="Plate confidence threshold")
    parser.add_argument("--ocr-conf", type=float, default=0.50, help="OCR confidence threshold")
    parser.add_argument("--ocr-interval", type=int, default=5, help="OCR interval in frames")
    parser.add_argument("--reid-interval", type=int, default=10, help="Re-ID interval in frames")
    parser.add_argument("--min-valid-obs", type=int, default=3, help="Min valid obs for stable plate")
    parser.add_argument("--min-stability", type=float, default=0.70, help="Min stability score for stable plate")
    parser.add_argument("--output-dir", type=str, default="runs/pipeline", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames to process")

    args = parser.parse_args()

    pipeline = UnifiedVehiclePipeline(
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        reid_weights_path=args.reid_weights,
        tracker_config_path=args.tracker_config,
        vehicle_conf=args.vehicle_conf,
        plate_conf=args.plate_conf,
        ocr_conf=args.ocr_conf,
        ocr_interval=args.ocr_interval,
        reid_interval=args.reid_interval,
        min_valid_observations=args.min_valid_obs,
        min_stability_score=args.min_stability,
        output_dir=args.output_dir,
    )

    results = pipeline.process_video(
        video_source=args.source,
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 70)
    print("NETRA UNIFIED VEHICLE PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    stats = results["summary_statistics"]
    meta = results["metadata"]
    print(f"Video Source:             {meta['source']}")
    print(f"Processed Frames:         {meta['processed_frames']} / {meta['total_frames']}")
    print(f"Processing Speed:         {meta['pipeline_fps']:.1f} FPS (Total: {meta['total_processing_time_sec']:.1f}s)")
    print(f"Total Vehicle Tracks:     {stats['total_vehicle_tracks']}")
    print(f"Tracks with Re-ID:        {stats['tracks_with_reid']}")
    print(f"Total Re-ID Observations: {stats['total_reid_observations']}")
    print(f"Avg Re-ID Latency:        {stats['average_reid_latency_ms']:.1f} ms")
    print(f"Plate Detections:         {stats['total_plate_detections']}")
    print(f"Plate Associations:       {stats['total_plate_associations']}")
    print(f"OCR Invocations:          {stats['total_ocr_attempts']}")
    print(f"Successful OCR Reads:     {stats['successful_ocr_reads']}")
    print(f"Avg OCR Latency:          {stats['average_ocr_latency_ms']:.1f} ms")
    print(f"Stable Plates:            {stats['stable_plate_tracks']}")
    print(f"Tentative Plates:         {stats['tentative_plate_tracks']}")
    print(f"Unknown Tracks:           {stats['unknown_plate_tracks']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
