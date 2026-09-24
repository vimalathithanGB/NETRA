"""
SIH 2026 NETRA AI ENGINE - Phase 5B: ByteTrack + Vehicle Re-ID Integration
=============================================================================
Module: tracking/reid_tracker.py

Description:
    First integration checkpoint between Single-Camera ByteTrack Multi-Object Tracking
    and the OSNet-AIN VeRi-776 Vehicle Re-Identification (Vehicle Re-ID) extractor.

Key Architecture Principles:
    1. Responsibility Separation:
       - ByteTrack is responsible for LOCAL single-camera tracking (Track IDs).
       - VehicleReIDExtractor is responsible for APPEARANCE EMBEDDINGS (512-D vectors).
       - This checkpoint does NOT perform cross-camera matching or assign global IDs.
    2. Single Re-ID Instance:
       - Exactly ONE VehicleReIDExtractor instance is initialized on the target device.
    3. Update Interval (Compute Throttling):
       - To protect 4 GB VRAM on the NVIDIA RTX 3050 Laptop GPU, embeddings are
         extracted upon track confirmation and then every N frames (default: 10).
    4. In-Memory Track Registry:
       - Track states are maintained purely in runtime memory (no DB writes).
    5. Unit Embedding Validation:
       - Every generated embedding is verified: shape == (512,), finite, L2-norm ~ 1.0.
    6. Crop Safety & Error Handling:
       - Bounding boxes are clamped; tiny or degenerate crops are safely rejected.
       - Extraction errors on a single crop do NOT halt video processing.
    7. Diagnostic Intra-Track Self-Consistency:
       - Computes cosine similarity between successive embeddings of the SAME local track
         to quantify temporal visual feature stability.

Usage:
    python tracking/reid_tracker.py --source data/videos/test_2.mp4 --no-display
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure project root directory is in sys.path for absolute package imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.vehicle_reid import VehicleReIDExtractor, compute_cosine_similarity


# =============================================================================
# 1. VISUALIZATION CONSTANTS & HELPERS
# =============================================================================

# Curated BGR Color Palette for 6 Target Classes
CLASS_COLORS = {
    0: (230, 160, 40),   # car           - Ocean Blue / Cyan
    1: (60, 220, 90),    # motorcycle    - Emerald Green
    2: (0, 215, 255),    # auto_rickshaw - Indian Yellow / Gold
    3: (40, 70, 230),    # bus           - Crimson Red
    4: (220, 70, 220),   # truck         - Magenta / Purple
    5: (180, 200, 50),   # van           - Lime Turquoise
}

DEFAULT_BOX_COLOR = (200, 200, 200)

MIN_CROP_WIDTH = 16
MIN_CROP_HEIGHT = 16


def get_track_color(track_id: int):
    """
    Generates a deterministic BGR color based on track ID.
    """
    golden_ratio = 0.618033988749895
    h = int(((track_id * golden_ratio) % 1.0) * 180)
    hsv_pixel = np.uint8([[[h, 200, 230]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def validate_embedding(emb: np.ndarray) -> bool:
    """
    Strict validation of generated Re-ID feature embedding:
    1. Is a numpy ndarray
    2. Shape is exactly (512,)
    3. Contains no NaN or Inf values
    4. Has L2 Euclidean norm approximately 1.0
    """
    if not isinstance(emb, np.ndarray):
        return False
    if emb.shape != (512,):
        return False
    if not np.all(np.isfinite(emb)):
        return False
    norm = float(np.linalg.norm(emb))
    if abs(norm - 1.0) > 1e-2:  # Allow slight float precision variance
        return False
    return True


# =============================================================================
# 2. FRAME ANNOTATION & RENDERING
# =============================================================================

def draw_reid_tracking_overlay(
    frame: np.ndarray,
    tracked_items: List[Dict[str, Any]],
    track_histories: Dict[int, deque],
    frame_idx: int,
    total_frames: int,
    current_fps: float,
    active_tracks_count: int,
    total_embeddings_count: int,
) -> np.ndarray:
    """
    Draws vehicle bounding boxes, Re-ID status tags, motion trails, and the NETRA HUD.
    """
    annotated = frame.copy()
    h_img, w_img = annotated.shape[:2]

    # Dynamic scaling for high-resolution video (e.g. 2560x1440)
    scale_factor = max(1.0, w_img / 1920.0)
    font_scale = 0.52 * scale_factor
    font_thickness = max(1, int(1.5 * scale_factor))
    box_thickness = max(2, int(2.0 * scale_factor))

    # 1. Draw motion trajectory trails
    for track_id, points in track_histories.items():
        if len(points) < 2:
            continue
        trail_color = get_track_color(track_id)
        pts = list(points)
        for i in range(1, len(pts)):
            t_thick = int(max(1, (i / len(pts)) * (3 * scale_factor)))
            cv2.line(annotated, pts[i - 1], pts[i], trail_color, t_thick, cv2.LINE_AA)

    # 2. Draw bounding boxes and Re-ID status banners
    for item in tracked_items:
        x1, y1, x2, y2 = item["bbox"]
        track_id = item["track_id"]
        class_name = item["class_name"].upper()
        conf = item["conf"]
        reid_status = item["reid_status"]  # "REID:OK" or "REID:WAIT"

        box_color = CLASS_COLORS.get(item["class_id"], get_track_color(track_id))

        # Draw main vehicle bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness, cv2.LINE_AA)

        # Build overlay text: e.g. "CAR | ID:17 | 0.91 | REID:OK"
        label_text = f"{class_name} | ID:{track_id} | {conf:.2f} | {reid_status}"

        # Calculate banner dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        (label_w, label_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        label_y1 = max(0, y1 - label_h - baseline - int(8 * scale_factor))
        label_y2 = label_y1 + label_h + baseline + int(8 * scale_factor)
        label_x2 = min(w_img, x1 + label_w + int(10 * scale_factor))

        # Status indicator banner background
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), box_color, -1)

        # Contrasting text
        brightness = box_color[0] * 0.299 + box_color[1] * 0.587 + box_color[2] * 0.114
        text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

        cv2.putText(
            annotated,
            label_text,
            (x1 + int(5 * scale_factor), label_y2 - baseline - int(4 * scale_factor)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    # 3. Draw NETRA Heads-Up Display (HUD)
    hud_h = int(50 * scale_factor)
    hud_w = min(w_img, int(820 * scale_factor))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)

    prog_str = f"F:{frame_idx:05d}/{total_frames:05d}" if total_frames > 0 else f"F:{frame_idx:05d}"
    hud_line1 = "NETRA -- BYTE TRACK + VEHICLE RE-ID"
    hud_line2 = (
        f"Frame: {prog_str} | Active Tracks: {active_tracks_count:02d} | "
        f"Embeddings: {total_embeddings_count:03d} | {current_fps:4.1f} FPS"
    )

    hud_font_scale = 0.50 * scale_factor
    cv2.putText(
        annotated,
        hud_line1,
        (int(10 * scale_factor), int(20 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        hud_font_scale,
        (0, 255, 200),
        font_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        hud_line2,
        (int(10 * scale_factor), int(42 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        hud_font_scale,
        (255, 255, 255),
        max(1, font_thickness - 1),
        cv2.LINE_AA,
    )

    return annotated


# =============================================================================
# 3. INTEGRATION PIPELINE CORE
# =============================================================================

def run_reid_tracking(
    source_path: str,
    model_path: str = "runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/best.pt",
    tracker_config_path: str = "tracking/tracker_config.yaml",
    conf_thresh: float = 0.40,
    imgsz: int = 640,
    device: str = "0",
    output_path: str = "runs/tracking/traffic_test_bytetrack_reid.mp4",
    reid_interval: int = 10,
    no_display: bool = True,
):
    """
    Runs unified ByteTrack tracking + OSNet-AIN Vehicle Re-ID feature extraction.
    """
    print("=" * 78)
    print("NETRA -- BYTE TRACK + VEHICLE RE-ID INTEGRATION PIPELINE")
    print("=" * 78)

    # 1. Verify Model Path
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"YOLO detector weights missing at: {model_file}")

    # 2. Verify Tracker Configuration
    tracker_file = Path(tracker_config_path).resolve()
    if tracker_file.exists():
        tracker_arg = str(tracker_file)
        tracker_desc = f"Custom ({tracker_file.name})"
    else:
        tracker_arg = "bytetrack.yaml"
        tracker_desc = "Ultralytics Default (bytetrack.yaml)"

    # 3. Verify Video Source
    src_file = Path(source_path).resolve()
    if not src_file.exists():
        raise FileNotFoundError(f"Input video source missing at: {src_file}")

    # 4. Resolve Device
    dev_str = str(device).strip()
    if dev_str.isdigit():
        reid_device = f"cuda:{dev_str}" if torch.cuda.is_available() else "cpu"
    elif dev_str == "cuda":
        reid_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        reid_device = dev_str

    if reid_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA device requested but torch.cuda is unavailable. Falling back to CPU.")
        reid_device = "cpu"
    device_name = torch.cuda.get_device_name(0) if reid_device != "cpu" else "CPU"

    print(f"  Input Video Source     : {src_file}")
    print(f"  Vehicle Detector       : {model_file} ({model_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Tracker Algorithm      : ByteTrack [{tracker_desc}]")
    print(f"  Re-ID Embedding Engine : OSNet-AIN x1.0 (VeRi-776, 512-D L2-normalized)")
    print(f"  Re-ID Extraction Cadence: Every {reid_interval} frames per track")
    print(f"  Confidence Threshold   : {conf_thresh:.2f}")
    print(f"  Detector Resolution    : {imgsz}x{imgsz}")
    print(f"  Compute Device         : {device} ({device_name})")
    print(f"  Display Window         : {'Disabled (--no-display)' if no_display else 'Interactive GUI'}")

    # 5. Open Input Video Capture
    cap = cv2.VideoCapture(str(src_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {src_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0.0 or video_fps > 120.0:
        video_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video Metadata         : {width}x{height} @ {video_fps:.2f} FPS ({total_frames:,} total frames)")

    # 6. Initialize Output Video Writer
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, video_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to initialize VideoWriter at: {out_file}")
    print(f"  Output Video Path      : {out_file}")
    print("-" * 78)

    # 7. Initialize Exactly ONE YOLO Detector and ONE VehicleReIDExtractor
    print("[*] Initializing YOLOv8n detector...")
    detector = YOLO(str(model_file))
    model_names = detector.names
    print(f"[OK] YOLOv8n detector loaded ({len(model_names)} classes).")

    print("[*] Initializing OSNet-AIN Vehicle Re-ID Extractor...")
    reid_extractor = VehicleReIDExtractor(device=reid_device)
    print(f"[OK] VehicleReIDExtractor loaded on device: {reid_extractor.device_str}.")
    print("-" * 78)

    # 8. In-Memory Runtime Track Registry & Telemetry State
    # track_id -> {
    #     "class_id": int,
    #     "class_name": str,
    #     "latest_embedding": np.ndarray (512,),
    #     "embedding_count": int,
    #     "last_reid_frame": int,
    #     "last_seen_frame": int,
    #     "last_seen_timestamp": float,
    #     "self_consistencies": list of float
    # }
    track_registry: Dict[int, Dict[str, Any]] = {}
    track_histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
    all_self_consistencies: List[float] = []

    total_detections = 0
    embeddings_generated = 0
    embedding_failures = 0
    invalid_crops_skipped = 0
    processed_frames = 0
    start_time = time.time()
    last_hud_fps = video_fps

    print("[*] Starting unified frame-by-frame tracking & Re-ID extraction...")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            processed_frames += 1
            timestamp = processed_frames / video_fps

            # -----------------------------------------------------------------
            # Step A: YOLO Detection + ByteTrack Multi-Object Tracking
            # -----------------------------------------------------------------
            results = detector.track(
                source=frame,
                persist=True,
                tracker=tracker_arg,
                conf=conf_thresh,
                imgsz=imgsz,
                device=reid_device,
                verbose=False,
            )

            tracked_items_for_frame = []
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                has_ids = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_ids else [-1] * len(boxes)

                for box_idx in range(len(boxes)):
                    total_detections += 1
                    raw_x1, raw_y1, raw_x2, raw_y2 = map(int, xyxy[box_idx])
                    cid = int(classes[box_idx])
                    conf = float(confs[box_idx])
                    tid = int(track_ids[box_idx])
                    cname = model_names.get(cid, f"class_{cid}")

                    # Ignore untracked background/ephemeral detections
                    if tid <= 0:
                        continue

                    # Record centroid for trajectory tail
                    cx = int((raw_x1 + raw_x2) / 2.0)
                    cy = int((raw_y1 + raw_y2) / 2.0)
                    track_histories[tid].append((cx, cy))

                    # Initialize track registry record if newly observed
                    if tid not in track_registry:
                        track_registry[tid] = {
                            "class_id": cid,
                            "class_name": cname,
                            "latest_embedding": None,
                            "embedding_count": 0,
                            "last_reid_frame": -9999,
                            "last_seen_frame": processed_frames,
                            "last_seen_timestamp": timestamp,
                            "self_consistencies": [],
                        }
                    else:
                        track_registry[tid]["last_seen_frame"] = processed_frames
                        track_registry[tid]["last_seen_timestamp"] = timestamp
                        track_registry[tid]["class_id"] = cid
                        track_registry[tid]["class_name"] = cname

                    # ---------------------------------------------------------
                    # Step B: Check Re-ID Update Interval Cadence
                    # ---------------------------------------------------------
                    last_reid = track_registry[tid]["last_reid_frame"]
                    needs_reid = (
                        track_registry[tid]["embedding_count"] == 0
                        or (processed_frames - last_reid) >= reid_interval
                    )

                    reid_status_str = (
                        "REID:OK" if track_registry[tid]["embedding_count"] > 0 else "REID:WAIT"
                    )

                    # ---------------------------------------------------------
                    # Step C: Crop Bounding Box & Extract Re-ID Feature Vector
                    # ---------------------------------------------------------
                    if needs_reid:
                        # 1. Clamp bounding box safely to frame boundaries
                        x1 = max(0, min(width - 1, raw_x1))
                        y1 = max(0, min(height - 1, raw_y1))
                        x2 = max(0, min(width, raw_x2))
                        y2 = max(0, min(height, raw_y2))
                        crop_w = x2 - x1
                        crop_h = y2 - y1

                        # 2. Reject invalid or tiny crops
                        if crop_w < MIN_CROP_WIDTH or crop_h < MIN_CROP_HEIGHT:
                            invalid_crops_skipped += 1
                        else:
                            crop = frame[y1:y2, x1:x2]
                            try:
                                # Extract 512-dim embedding
                                embedding = reid_extractor.extract_embedding(crop)

                                # 3. Strict embedding validation
                                if validate_embedding(embedding):
                                    embeddings_generated += 1

                                    # 4. Optional diagnostic self-consistency check
                                    prev_emb = track_registry[tid]["latest_embedding"]
                                    if prev_emb is not None:
                                        sim = compute_cosine_similarity(prev_emb, embedding)
                                        track_registry[tid]["self_consistencies"].append(sim)
                                        all_self_consistencies.append(sim)

                                    # Update track registry
                                    track_registry[tid]["latest_embedding"] = embedding
                                    track_registry[tid]["embedding_count"] += 1
                                    track_registry[tid]["last_reid_frame"] = processed_frames
                                    reid_status_str = "REID:OK"
                                else:
                                    embedding_failures += 1
                                    reid_status_str = "REID:WAIT"

                            except Exception as reid_err:
                                # Safe error handling: single bad crop does NOT crash video
                                embedding_failures += 1
                                reid_status_str = "REID:WAIT"

                    tracked_items_for_frame.append({
                        "bbox": (raw_x1, raw_y1, raw_x2, raw_y2),
                        "track_id": tid,
                        "class_id": cid,
                        "class_name": cname,
                        "conf": conf,
                        "reid_status": reid_status_str,
                    })

            # Calculate processing speed
            dt = time.perf_counter() - t0
            instant_fps = 1.0 / dt if dt > 0 else video_fps
            last_hud_fps = 0.9 * last_hud_fps + 0.1 * instant_fps

            # -----------------------------------------------------------------
            # Step D: Render Overlays & Write Video
            # -----------------------------------------------------------------
            annotated_frame = draw_reid_tracking_overlay(
                frame=frame,
                tracked_items=tracked_items_for_frame,
                track_histories=track_histories,
                frame_idx=processed_frames,
                total_frames=total_frames,
                current_fps=last_hud_fps,
                active_tracks_count=len(tracked_items_for_frame),
                total_embeddings_count=embeddings_generated,
            )

            writer.write(annotated_frame)

            if not no_display:
                cv2.imshow("NETRA - ByteTrack + Vehicle Re-ID Checkpoint", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[!] User terminated interactive preview early.")
                    break

            # Progress logging
            if processed_frames % 20 == 0 or processed_frames == total_frames:
                pct = (processed_frames / total_frames * 100.0) if total_frames > 0 else 0.0
                sys.stdout.write(
                    f"\r--> Frame {processed_frames:4d}/{total_frames:4d} ({pct:5.1f}%) | "
                    f"Speed: {last_hud_fps:5.1f} FPS | "
                    f"Active: {len(tracked_items_for_frame):2d} | "
                    f"Embeddings: {embeddings_generated:3d} (Fail: {embedding_failures}) | "
                    f"Unique Tracks: {len(track_registry):2d}"
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user (Ctrl+C). Finalizing outputs safely...")
    finally:
        cap.release()
        writer.release()
        if not no_display:
            cv2.destroyAllWindows()

    total_elapsed = time.time() - start_time
    avg_fps = processed_frames / total_elapsed if total_elapsed > 0 else 0.0

    # -------------------------------------------------------------------------
    # Step E: Comprehensive End-of-Run Telemetry Report
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 50)
    print("NETRA BYTE TRACK + VEHICLE RE-ID TEST")
    print("=" * 50)
    print(f"Video: {src_file.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {video_fps:.2f}")
    print(f"Frames processed: {processed_frames:,}")
    print()
    print(f"Unique Track IDs: {len(track_registry):,}")
    print(f"Total detections: {total_detections:,}")
    print()
    print(f"Embeddings generated: {embeddings_generated:,}")
    print(f"Embedding failures: {embedding_failures:,}")
    print(f"Invalid crops skipped: {invalid_crops_skipped:,}")
    print()
    print("Active/final track states:")
    print("Track ID | Class           | Embeddings | Last Frame")
    print("-" * 52)
    for tid in sorted(track_registry.keys()):
        tinfo = track_registry[tid]
        cname = tinfo["class_name"]
        cnt = tinfo["embedding_count"]
        last_f = tinfo["last_seen_frame"]
        print(f"ID:{tid:<5} | {cname:<15} | {cnt:<10} | Frame {last_f}")
    print()
    print(f"Average processing FPS: {avg_fps:.2f}")

    if all_self_consistencies:
        arr = np.array(all_self_consistencies, dtype=np.float32)
        print("\n" + "=" * 50)
        print("EMBEDDING SELF-CONSISTENCY SUMMARY (Same-Track Diagnostic)")
        print("=" * 50)
        tracks_with_multi = sum(1 for t in track_registry.values() if t["embedding_count"] > 1)
        print(f"Tracks with >= 2 embeddings : {tracks_with_multi}")
        print(f"Total intra-track updates   : {len(arr)}")
        print(f"Mean Cosine Similarity       : {float(np.mean(arr)):.4f}")
        print(f"Std  Cosine Similarity       : {float(np.std(arr)):.4f}")
        print(f"Min  Cosine Similarity       : {float(np.min(arr)):.4f}")
        print(f"Max  Cosine Similarity       : {float(np.max(arr)):.4f}")
        print("Note: High cosine similarity (typically > 0.85) confirms stable visual appearance")
        print("representation across camera perspective shifts and scale changes.")
    else:
        print("\n[NOTE] No multi-embedding tracks were long enough to compute intra-track consistency.")

    print("=" * 50)
    print(f"[OK] Annotated output video generated at:\n     {out_file}")
    print("=" * 50)


# =============================================================================
# 4. COMMAND LINE PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH 2026 NETRA - Phase 5B: ByteTrack + Vehicle Re-ID Integration Checkpoint"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/videos/test_2.mp4",
        help="Path to input video file (default: data/videos/test_2.mp4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/best.pt",
        help="Path to fine-tuned YOLOv8 vehicle detection weights.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="tracking/tracker_config.yaml",
        help="Path to tracker YAML configuration file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold (default: 0.40).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: '0' for CUDA GPU, 'cpu' for fallback).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/tracking/traffic_test_bytetrack_reid.mp4",
        help="Path to output annotated MP4 video file.",
    )
    parser.add_argument(
        "--reid-interval",
        type=int,
        default=10,
        help="Embedding update cadence in frames per track (default: 10).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=True,
        help="Run headless without opening OpenCV GUI display window (default: True).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_reid_tracking(
        source_path=cli_args.source,
        model_path=cli_args.model,
        tracker_config_path=cli_args.tracker,
        conf_thresh=cli_args.conf,
        imgsz=cli_args.imgsz,
        device=cli_args.device,
        output_path=cli_args.output,
        reid_interval=cli_args.reid_interval,
        no_display=cli_args.no_display,
    )
