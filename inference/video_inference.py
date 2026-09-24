"""
SIH 2026 AI Engine - Phase 2: Pretrained YOLO Video Inference
============================================================
Module: inference/video_inference.py

Description:
    Runs local frame-by-frame vehicle detection on video files using a lightweight,
    pretrained YOLOv8 model (yolov8n.pt) optimized for 4 GB VRAM GPUs (NVIDIA RTX 3050).

Key Features:
    - Processes video frames sequentially via OpenCV.
    - Runs accelerated CUDA inference on each frame.
    - Renders bounding boxes and confidence tags using result.plot().
    - Encodes and saves the resulting annotated video into data/outputs/.
    - Computes and displays real-time processing FPS.
    - No tracking algorithms are integrated yet (reserved for Phase 4).

Usage:
    python inference/video_inference.py --source data/videos/test.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import torch
from ultralytics import YOLO

# Vehicle classes present in the generic COCO dataset
TARGET_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for video inference."""
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLO vehicle detection on a video stream or file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/videos/test.mp4",
        help="Path to the input video file (e.g., data/videos/test.mp4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights file (e.g., yolov8n.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Directory to save the annotated video.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence detection threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: 'cuda', 'cuda:0', 'cpu', or leave empty for auto-detection.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="If specified, detect all 80 COCO classes instead of only vehicles.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = entire video). Useful for quick benchmark tests.",
    )
    return parser.parse_args()


def select_device(preferred_device: str = "") -> str:
    """Select compute device (CUDA GPU or CPU)."""
    if preferred_device:
        return preferred_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def print_banner():
    """Display an educational header for beginner developers."""
    print("=" * 70)
    print("  SIH 2026 AI ENGINE - PHASE 2: PRETRAINED YOLO VIDEO INFERENCE")
    print("=" * 70)


def print_educational_note():
    """Clarify the capabilities of generic COCO pretrained models on video."""
    print("-" * 70)
    print(" [NOTE - ARCHITECTURAL BASELINE]")
    print(" This script runs frame-by-frame vehicle detection on video footage.")
    print(" - Frame tracking (ByteTrack / BoT-SORT) is intentionally NOT enabled yet.")
    print(" - Bounding boxes are recomputed independently for each frame.")
    print(" - Tracking and persistent vehicle IDs will be added in Phase 4.")
    print("-" * 70)


def run_video_inference(
    source_path: str,
    model_name: str = "yolov8n.pt",
    output_dir: str = "data/outputs",
    conf_thresh: float = 0.25,
    device_arg: str = "",
    all_classes: bool = False,
    max_frames: int = 0,
) -> Path:
    """
    Execute YOLO inference frame-by-frame on a video file.

    Args:
        source_path: Path to the input video.
        model_name: Name or path to the YOLO weights.
        output_dir: Output folder for annotated video.
        conf_thresh: Minimum confidence score to retain detections.
        device_arg: User-specified compute device.
        all_classes: Whether to detect all COCO classes or only target vehicles.
        max_frames: Max frames to process (0 = process full video).

    Returns:
        Path to the saved annotated MP4 video.
    """
    input_path = Path(source_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Video file not found: '{source_path}'\n"
            f"Please place your video file inside 'data/videos/' or pass a valid path using:\n"
            f"  python inference/video_inference.py --source <path_to_video>"
        )

    # 1. Resolve compute device
    device = select_device(device_arg)
    gpu_name = torch.cuda.get_device_name(0) if "cuda" in device and torch.cuda.is_available() else "N/A"

    print(f"\n[1/5] Hardware & Model Setup:")
    print(f"  - Model File     : {model_name}")
    print(f"  - Selected Device: {device}")
    if gpu_name != "N/A":
        print(f"  - GPU Name       : {gpu_name}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  - Total VRAM     : {vram_gb:.2f} GB (RTX 3050 friendly)")

    # 2. Load model
    model_path = Path(model_name)
    if not model_path.exists() and Path("models", model_name).exists():
        model_path = Path("models", model_name)

    print(f"[2/5] Loading YOLO weights from: {model_path} ...")
    model = YOLO(str(model_path))

    # Identify target vehicle classes
    all_names: Dict[int, str] = model.names
    target_class_ids = [
        cls_id for cls_id, name in all_names.items() if name in TARGET_VEHICLE_CLASSES
    ]
    filter_classes = None if all_classes else target_class_ids

    # 3. Open Video Source via OpenCV
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or input_fps > 120:
        input_fps = 30.0  # Fallback to standard 30 FPS if header is corrupted
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_process = total_frames if max_frames <= 0 else min(total_frames, max_frames)

    print(f"[3/5] Video Stream Properties:")
    print(f"  - Source Path    : {input_path.resolve()}")
    print(f"  - Resolution     : {width} x {height} px")
    print(f"  - Native FPS     : {input_fps:.2f}")
    print(f"  - Total Frames   : {total_frames} (processing up to {frames_to_process})")
    print(f"  - Target Classes : {'ALL COCO CLASSES' if all_classes else list(TARGET_VEHICLE_CLASSES)}")
    print(f"  - Confidence Cut : >= {conf_thresh:.2f}")

    # 4. Prepare VideoWriter for Annotated Output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{input_path.stem}_annotated.mp4"
    output_filepath = out_dir / output_filename

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_filepath),
        fourcc,
        input_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer for output: {output_filepath}")

    # 5. Process Frames
    print(f"[4/5] Running frame-by-frame inference on {device} ...")
    processed_count = 0
    total_detections_sum = 0
    start_total_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_count += 1
            frame_start = time.perf_counter()

            # Run inference on single frame
            results = model.predict(
                source=frame,
                conf=conf_thresh,
                classes=filter_classes,
                device=device,
                verbose=False,
            )
            frame_inference_time = time.perf_counter() - frame_start

            result = results[0]
            num_dets = len(result.boxes) if result.boxes is not None else 0
            total_detections_sum += num_dets

            # Render annotations onto frame
            annotated_frame = result.plot()
            writer.write(annotated_frame)

            # Calculate rolling performance
            fps = 1.0 / frame_inference_time if frame_inference_time > 0 else 0.0
            percent = (processed_count / frames_to_process * 100.0) if frames_to_process > 0 else 0.0

            # Progress log (printed every 10 frames or on last frame)
            if processed_count % 10 == 0 or processed_count == frames_to_process:
                print(
                    f"  Frame [{processed_count:>4}/{frames_to_process:>4}] "
                    f"({percent:>5.1f}%) | Speed: {fps:>5.1f} FPS | "
                    f"Detections in frame: {num_dets:<2}"
                )

            if max_frames > 0 and processed_count >= max_frames:
                print(f"  Reached max-frames limit ({max_frames}). Stopping inference.")
                break

    finally:
        cap.release()
        writer.release()

    total_duration = time.perf_counter() - start_total_time
    avg_fps = processed_count / total_duration if total_duration > 0 else 0.0

    # 6. Detailed Console Summary
    print("\n" + "=" * 70)
    print("                  VIDEO INFERENCE SUMMARY REPORT")
    print("=" * 70)
    print(f" Model Used            : {model_name} (YOLOv8 Nano)")
    print(f" Compute Device        : {device} ({gpu_name})")
    print(f" Input Video           : {input_path}")
    print(f" Resolution            : {width} x {height} px @ {input_fps:.2f} FPS")
    print(f" Frames Processed      : {processed_count} / {total_frames}")
    print(f" Total Elapsed Time    : {total_duration:.2f} seconds")
    print(f" Average Processing FPS: {avg_fps:.1f} FPS")
    print(f" Total Detections Made : {total_detections_sum}")
    if processed_count > 0:
        print(f" Avg Detections/Frame  : {total_detections_sum / processed_count:.1f}")
    print("-" * 70)
    print(f" Output Video Path     : {output_filepath.resolve()}")
    print("=" * 70)

    return output_filepath


def main():
    args = parse_arguments()
    print_banner()
    print_educational_note()

    try:
        output_file = run_video_inference(
            source_path=args.source,
            model_name=args.model,
            output_dir=args.output_dir,
            conf_thresh=args.conf,
            device_arg=args.device,
            all_classes=args.all_classes,
            max_frames=args.max_frames,
        )
        print(f"\n[SUCCESS] Video inference complete! Output saved to: {output_file}\n")
    except FileNotFoundError as fnf_err:
        print(f"\n[WARNING] {fnf_err}", file=sys.stderr)
        print("\nTip: To test video inference, copy any sample MP4 into 'data/videos/test.mp4' and run again.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Video inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
