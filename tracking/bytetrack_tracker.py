"""
SIH 2026 AI Engine - Phase 5A: Single-Camera Vehicle Tracking with ByteTrack
=============================================================================
Module: tracking/bytetrack_tracker.py

Description:
    Performs real-time frame-by-frame vehicle detection and multi-object tracking
    (MOT) using the fine-tuned YOLOv8n UVH-26 detector paired with the ByteTrack
    association algorithm.

Target Classes (6):
    0: car
    1: motorcycle
    2: auto_rickshaw
    3: bus
    4: truck
    5: van

Key Capabilities:
    1. Loads fine-tuned Model 1: runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/best.pt
    2. Runs frame-by-frame YOLOv8 detection + ByteTrack motion association (persist=True).
    3. Employs 2-stage IoU and Kalman filter tracking (recovering occluded vehicles).
    4. Renders standardized surveillance overlays:
       - Distinct class & track-colored bounding boxes
       - Tracking banner: <class_name> | ID:<track_id> | <conf>
       - Smooth motion trajectory tail (last 30 centroid points)
       - Real-time diagnostic HUD (FPS, active track count, frame progress)
    5. Saves annotated MP4 output to runs/tracking/ (preserving original video FPS/resolution).
    6. Produces comprehensive end-of-run telemetry and class-level tracking analytics.
    7. Supports headless execution via --no-display (default for server/pipeline mode).
    8. Optimized for 4 GB VRAM NVIDIA RTX 3050 Laptop GPU (stream-based memory management).

Usage Examples:
    # Basic tracking test on default sample video:
    python tracking/bytetrack_tracker.py --source data/videos/test.mp4 --no-display

    # Custom video with specific confidence threshold:
    python tracking/bytetrack_tracker.py --source data/videos/traffic_cctv1.mp4 --conf 0.40 --no-display

    # Interactive visualization window (press 'q' to exit):
    python tracking/bytetrack_tracker.py --source data/videos/test.mp4
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
import cv2
import torch
from ultralytics import YOLO


# =============================================================================
# 1. VISUALIZATION & COLOR PALETTE
# =============================================================================

# Curated BGR Color Palette for 6 Target Vehicle Classes
CLASS_COLORS = {
    0: (230, 160, 40),   # car           - Ocean Blue / Cyan
    1: (60, 220, 90),    # motorcycle    - Emerald Green
    2: (0, 215, 255),    # auto_rickshaw - Indian Yellow / Gold
    3: (40, 70, 230),    # bus           - Crimson Red
    4: (220, 70, 220),   # truck         - Magenta / Purple
    5: (180, 200, 50),   # van           - Lime Turquoise
}

# Fallback color for unknown classes
DEFAULT_COLOR = (200, 200, 200)

def get_track_color(track_id: int):
    """
    Generates a deterministic, aesthetically pleasing BGR color based on track ID.
    Ensures adjacent IDs have contrasting hues.
    """
    golden_ratio = 0.618033988749895
    h = int(((track_id * golden_ratio) % 1.0) * 180)
    # Convert HSV to BGR using OpenCV
    import numpy as np
    hsv_pixel = np.uint8([[[h, 200, 230]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


# =============================================================================
# 2. FRAME ANNOTATION & RENDERING
# =============================================================================

def draw_tracking_overlay(
    frame,
    tracked_boxes,
    track_histories,
    model_names,
    frame_idx,
    total_frames,
    current_fps,
    unique_tracks_count
):
    """
    Draws professional bounding boxes, tracking ID banners, and motion trails.
    """
    annotated = frame.copy()
    h_img, w_img = annotated.shape[:2]

    # 1. Draw motion trajectory trails (centroids history)
    for track_id, points in track_histories.items():
        if len(points) < 2:
            continue
        trail_color = get_track_color(track_id)
        pts = list(points)
        for i in range(1, len(pts)):
            thickness = int(max(1, (i / len(pts)) * 3))
            cv2.line(annotated, pts[i - 1], pts[i], trail_color, thickness, cv2.LINE_AA)

    # 2. Draw bounding boxes and labels
    for item in tracked_boxes:
        x1, y1, x2, y2 = item["bbox"]
        track_id = item["track_id"]
        class_id = item["class_id"]
        conf = item["conf"]
        class_name = model_names.get(class_id, f"class_{class_id}")

        box_color = CLASS_COLORS.get(class_id, get_track_color(track_id))

        # Draw main bounding box with rounded aesthetic
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Build label: e.g. "car | ID:12 | 0.87"
        label_text = f"{class_name} | ID:{track_id} | {conf:.2f}"

        # Calculate label size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Ensure label banner fits within image bounds
        label_y1 = max(0, y1 - label_h - baseline - 6)
        label_y2 = label_y1 + label_h + baseline + 6
        label_x2 = min(w_img, x1 + label_w + 10)

        # Draw solid background for high text legibility
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), box_color, -1)

        # Draw dark/white contrasting text
        text_color = (0, 0, 0) if (box_color[0]*0.299 + box_color[1]*0.587 + box_color[2]*0.114) > 140 else (255, 255, 255)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 5, label_y2 - baseline - 3),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    # 3. Draw Top-Left Diagnostics Heads-Up Display (HUD)
    hud_h = 38
    hud_w = min(w_img, 460)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    prog_str = f"F:{frame_idx}/{total_frames}" if total_frames > 0 else f"F:{frame_idx}"
    hud_text = f"SIH ByteTrack | {prog_str} | {current_fps:.1f} FPS | Tracks: {unique_tracks_count}"
    cv2.putText(
        annotated,
        hud_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 200),
        1,
        cv2.LINE_AA
    )

    return annotated


# =============================================================================
# 3. TRACKING PIPELINE CORE
# =============================================================================

def run_bytetrack_tracking(
    source_path: str,
    model_path: str,
    tracker_config_path: str,
    conf_thresh: float = 0.40,
    imgsz: int = 640,
    device: str = "0",
    output_path: str = "runs/tracking/traffic_test_bytetrack.mp4",
    no_display: bool = True
):
    """
    Executes ByteTrack multi-object tracking on an input video source.
    """
    print("=" * 78)
    print("SIH 2026 AI ENGINE - VEHICLE TRACKING (YOLOv8 + ByteTrack)")
    print("=" * 78)

    # 1. Verify Model Path
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained vehicle detection model not found at:\n  {model_file}\n"
            "Please verify training completion or specify correct --model path."
        )

    # 2. Verify Tracker Configuration
    tracker_file = Path(tracker_config_path).resolve()
    if tracker_file.exists():
        tracker_arg = str(tracker_file)
        tracker_desc = f"Custom ({tracker_file.name})"
    else:
        # Fallback to Ultralytics built-in bytetrack.yaml
        tracker_arg = "bytetrack.yaml"
        tracker_desc = "Ultralytics Built-In (bytetrack.yaml)"

    # 3. Verify Video Source
    src_file = Path(source_path).resolve()
    if not src_file.exists():
        raise FileNotFoundError(
            f"Input video source not found at:\n  {src_file}\n"
            "Please check the --source path."
        )

    # 4. Resolve Device
    if str(device) != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA device requested but torch.cuda is unavailable. Falling back to CPU.")
        device = "cpu"

    print(f"  Input Video Source  : {src_file}")
    print(f"  Vehicle Detector    : {model_file} ({model_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Tracker Algorithm   : ByteTrack [{tracker_desc}]")
    print(f"  Confidence Cutoff   : {conf_thresh:.2f}")
    print(f"  Inference Size      : {imgsz}x{imgsz}")
    print(f"  Compute Device      : {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")
    print(f"  GUI Display Window  : {'Disabled (--no-display)' if no_display else 'Enabled'}")

    # 5. Open Input Video Capture
    cap = cv2.VideoCapture(str(src_file))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file: {src_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0.0 or video_fps > 120.0:
        video_fps = 30.0  # Fallback standard FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video Metadata      : {width}x{height} @ {video_fps:.2f} FPS ({total_frames:,} total frames)")

    # 6. Initialize Output Video Writer
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, video_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV failed to initialize VideoWriter at: {out_file}")
    print(f"  Output Video Path   : {out_file}")
    print("-" * 78)

    # 7. Load YOLO Vehicle Detector Model
    print("[*] Loading YOLOv8n detector weights into memory...")
    model = YOLO(str(model_file))
    model_names = model.names
    print(f"[OK] Model loaded. Recognized vehicle classes: {model_names}")
    print("-" * 78)

    # 8. Tracking State & Telemetry Collectors
    track_histories = defaultdict(lambda: deque(maxlen=30))
    all_unique_track_ids = set()
    tracks_by_class = defaultdict(set)       # class_name -> set of track_ids
    detections_by_class = defaultdict(int)   # class_name -> total bounding box count
    processed_frames = 0
    start_time = time.time()
    last_hud_fps = video_fps

    print("[*] Processing video stream frame-by-frame with ByteTrack...")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            processed_frames += 1

            # Run YOLO + ByteTrack on the frame
            # persist=True maintains Kalman filters & track states across frames
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_arg,
                conf=conf_thresh,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            # Parse detections and active track IDs
            tracked_boxes = []
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                has_ids = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_ids else [-1] * len(boxes)

                for box_idx in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, xyxy[box_idx])
                    cid = classes[box_idx]
                    conf = float(confs[box_idx])
                    tid = int(track_ids[box_idx])
                    cname = model_names.get(cid, str(cid))

                    detections_by_class[cname] += 1

                    if tid > 0:
                        all_unique_track_ids.add(tid)
                        tracks_by_class[cname].add(tid)

                        # Record centroid for motion trajectory trail
                        cx = int((x1 + x2) / 2.0)
                        cy = int((y1 + y2) / 2.0)
                        track_histories[tid].append((cx, cy))

                        tracked_boxes.append({
                            "bbox": (x1, y1, x2, y2),
                            "track_id": tid,
                            "class_id": cid,
                            "conf": conf
                        })

            # Calculate instantaneous processing FPS
            dt = time.perf_counter() - t0
            instant_fps = 1.0 / dt if dt > 0 else video_fps
            last_hud_fps = 0.9 * last_hud_fps + 0.1 * instant_fps

            # Render overlay annotations
            annotated_frame = draw_tracking_overlay(
                frame=frame,
                tracked_boxes=tracked_boxes,
                track_histories=track_histories,
                model_names=model_names,
                frame_idx=processed_frames,
                total_frames=total_frames,
                current_fps=last_hud_fps,
                unique_tracks_count=len(all_unique_track_ids)
            )

            # Write frame to output video
            writer.write(annotated_frame)

            # Interactive preview (if not --no-display)
            if not no_display:
                cv2.imshow("SIH 2026 AI Engine - ByteTrack Vehicle Tracker", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[!] User terminated tracking preview early (pressed 'q').")
                    break

            # Periodic console progress logging
            if processed_frames % 10 == 0 or processed_frames == total_frames:
                pct = (processed_frames / total_frames * 100.0) if total_frames > 0 else 0.0
                sys.stdout.write(
                    f"\r--> Frame {processed_frames:4d}/{total_frames:4d} ({pct:5.1f}%) | "
                    f"Speed: {last_hud_fps:5.1f} FPS | "
                    f"Active Tracks: {len(tracked_boxes):2d} | "
                    f"Cumulative Unique IDs: {len(all_unique_track_ids):3d}"
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[!] Tracking interrupted by user (Ctrl+C). Finalizing saved video...")
    finally:
        # Ensure resources are released cleanly under all circumstances
        cap.release()
        writer.release()
        if not no_display:
            cv2.destroyAllWindows()

    total_elapsed = time.time() - start_time
    avg_fps = processed_frames / total_elapsed if total_elapsed > 0 else 0.0

    print("\n" + "=" * 78)
    print("BYTE TRACK VEHICLE TRACKING TELEMETRY REPORT")
    print("=" * 78)
    print(f"  Input Video File        : {src_file}")
    print(f"  Output Annotated Video  : {out_file}")
    print(f"  Video Native Resolution : {width}x{height} @ {video_fps:.2f} FPS")
    print(f"  Total Video Frames      : {total_frames:,}")
    print(f"  Processed Frames Count  : {processed_frames:,}")
    print(f"  Total Tracking Duration : {total_elapsed:.2f}s ({avg_fps:.1f} avg processing FPS)")
    print(f"  Confidence Threshold    : {conf_thresh:.2f}")
    print(f"  Tracking Algorithm      : ByteTrack ({tracker_desc})")
    print(f"  Total Cumulative Detections : {sum(detections_by_class.values()):,}")
    print(f"  Total Unique Vehicle IDs    : {len(all_unique_track_ids):,}")
    print("-" * 78)
    print(f"  {'Vehicle Class':<16} {'Unique Track IDs':<20} {'Total Frame Detections':<24}")
    print("-" * 78)
    for cid in sorted(model_names.keys()):
        cname = model_names[cid]
        num_tracks = len(tracks_by_class[cname])
        num_dets = detections_by_class[cname]
        print(f"  {cname:<16} {num_tracks:<20,} {num_dets:<24,}")
    print("=" * 78)
    print(f"[SUCCESS] Tracked video output saved to: {out_file}")
    print("=" * 78)


# =============================================================================
# 4. CLI ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH 2026 AI Engine - Phase 5A: Single-Camera ByteTrack Vehicle Tracker."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/videos/test.mp4",
        help="Path to input video file (e.g. data/videos/traffic_test.mp4).",
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
        help="Path to tracker YAML configuration file (default: tracking/tracker_config.yaml).",
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
        default="runs/tracking/traffic_test_bytetrack.mp4",
        help="Path to output annotated MP4 video file.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Run headless without opening OpenCV GUI display window.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_bytetrack_tracking(
        source_path=cli_args.source,
        model_path=cli_args.model,
        tracker_config_path=cli_args.tracker,
        conf_thresh=cli_args.conf,
        imgsz=cli_args.imgsz,
        device=cli_args.device,
        output_path=cli_args.output,
        no_display=cli_args.no_display,
    )
