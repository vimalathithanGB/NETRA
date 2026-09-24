"""
SIH 2026 AI Engine - Phase 2: Pretrained YOLO Image Inference
============================================================
Module: inference/image_inference.py

Description:
    Runs local object detection inference on single images using a lightweight,
    pretrained YOLOv8 model (yolov8n.pt) optimized for 4 GB VRAM GPUs (NVIDIA RTX 3050).

Key Features:
    - Automatically utilizes CUDA GPU acceleration when available.
    - Filters and reports standard vehicle classes: car, motorcycle, bus, truck.
    - Saves annotated output images with bounding boxes, labels, and confidences.
    - Detailed console reporting of inference timings, device stats, and detections.

Usage:
    python inference/image_inference.py --source data/images/traffic_test.jpg
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from ultralytics import YOLO

# Standard vehicle classes present in the COCO dataset (80 classes)
# Pretrained YOLO models (like yolov8n.pt) are trained on the COCO dataset.
TARGET_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for image inference."""
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLO vehicle detection on a static image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images/traffic_test.jpg",
        help="Path to the input image file.",
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
        help="Directory where annotated output images will be saved.",
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
    return parser.parse_args()


def select_device(preferred_device: str = "") -> str:
    """
    Select the appropriate compute device (CUDA GPU or CPU).
    Validates CUDA availability and prints device metadata.
    """
    if preferred_device:
        device = preferred_device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return device


def print_banner():
    """Display an educational header for beginner developers."""
    print("=" * 70)
    print("  SIH 2026 AI ENGINE - PHASE 2: PRETRAINED YOLO INFERENCE")
    print("=" * 70)


def print_educational_note():
    """Clarify the capabilities of generic COCO pretrained models."""
    print("-" * 70)
    print(" [NOTE - ARCHITECTURAL BASELINE]")
    print(" This experiment uses a generic pretrained YOLO model (trained on COCO).")
    print(" It detects standard vehicle categories: [car, motorcycle, bus, truck].")
    print(" It is NOT trained to detect or read Indian license plates yet.")
    print(" Dedicated license plate localization and OCR models will be added")
    print(" in subsequent phases using custom Indian traffic datasets.")
    print("-" * 70)


def run_inference(
    source_path: str,
    model_name: str = "yolov8n.pt",
    output_dir: str = "data/outputs",
    conf_thresh: float = 0.25,
    device_arg: str = "",
    all_classes: bool = False,
) -> Path:
    """
    Execute YOLO inference on a single image and save the annotated result.

    Args:
        source_path: Path to the input image.
        model_name: Name or path to the YOLO weights.
        output_dir: Output folder for annotated results.
        conf_thresh: Minimum confidence score to retain detections.
        device_arg: User-specified compute device.
        all_classes: Whether to detect all COCO classes or only target vehicles.

    Returns:
        Path to the saved annotated image.
    """
    input_path = Path(source_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: '{source_path}'\n"
            f"Please verify that the file path is correct."
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

    # 2. Load the YOLO model
    # Look in 'models/' directory first if file exists there, otherwise root/cache
    model_path = Path(model_name)
    if not model_path.exists() and Path("models", model_name).exists():
        model_path = Path("models", model_name)

    print(f"[2/5] Loading YOLO weights from: {model_path} ...")
    model = YOLO(str(model_path))

    # Identify class IDs corresponding to target vehicles
    # COCO class mapping: 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
    all_names: Dict[int, str] = model.names
    target_class_ids = [
        cls_id for cls_id, name in all_names.items() if name in TARGET_VEHICLE_CLASSES
    ]

    filter_classes = None if all_classes else target_class_ids

    # 3. Read image dimensions
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise ValueError(f"OpenCV failed to read the image file: {input_path}")
    height, width, channels = img_bgr.shape

    print(f"[3/5] Processing Input Image:")
    print(f"  - Image Path     : {input_path.resolve()}")
    print(f"  - Dimensions     : {width} x {height} px ({channels} channels)")
    print(f"  - Target Classes : {'ALL COCO CLASSES' if all_classes else list(TARGET_VEHICLE_CLASSES)}")
    print(f"  - Confidence Cut : >= {conf_thresh:.2f}")

    # 4. Run Inference
    print(f"[4/5] Running YOLO inference on {device} ...")
    start_time = time.perf_counter()

    results = model.predict(
        source=str(input_path),
        conf=conf_thresh,
        classes=filter_classes,
        device=device,
        verbose=False,
    )
    total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    result = results[0]
    boxes = result.boxes

    # Extract Ultralytics internal latency measurements (in milliseconds)
    speed_info = getattr(result, "speed", {})
    inference_ms = speed_info.get("inference", 0.0)
    preprocess_ms = speed_info.get("preprocess", 0.0)
    postprocess_ms = speed_info.get("postprocess", 0.0)

    # Count detections by class
    detection_counts: Dict[str, int] = {}
    detailed_detections: List[Dict] = []

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0].item())
            cls_name = all_names.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0].item())
            coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]

            detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
            detailed_detections.append({
                "id": i + 1,
                "class": cls_name,
                "confidence": confidence,
                "bbox": coords,  # [x1, y1, x2, y2]
            })

    total_detections = len(detailed_detections)

    # 5. Save Annotated Output with unique filename containing confidence
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_tag = f"conf{conf_thresh:.2f}"
    base_name = f"{input_path.stem}_{conf_tag}_annotated"
    output_filename = f"{base_name}{input_path.suffix}"
    output_filepath = out_dir / output_filename

    # Ensure unique filename if repeated experiments are executed
    collision_idx = 1
    while output_filepath.exists():
        output_filename = f"{base_name}_{collision_idx}{input_path.suffix}"
        output_filepath = out_dir / output_filename
        collision_idx += 1

    # result.plot() renders bounding boxes, class labels, and confidence tags
    annotated_frame = result.plot()
    cv2.imwrite(str(output_filepath), annotated_frame)

    # 6. Detailed Console Summary (TASK 3 & PHASE 2.2)
    print("\n" + "=" * 70)
    print("                     INFERENCE SUMMARY REPORT")
    print("=" * 70)
    print(f" Model Used            : {model_name} (YOLOv8 Nano)")
    print(f" Compute Device        : {device} ({gpu_name})")
    print(f" Input Image           : {input_path}")
    print(f" Image Resolution      : {width} x {height} px")
    print(f" Confidence Threshold  : {conf_thresh:.2f} ({conf_thresh * 100:.0f}%)")
    print(f" Total Detections      : {total_detections}")
    print(f" Vehicle Counts        : ", end="")
    if detection_counts:
        counts_str = ", ".join([f"{k}: {v}" for k, v in detection_counts.items()])
        print(counts_str)
    else:
        print("None detected at confidence >= " + str(conf_thresh))

    print("-" * 70)
    print(f" Latency Timings:")
    print(f"  - Preprocessing      : {preprocess_ms:.2f} ms")
    print(f"  - Model Inference    : {inference_ms:.2f} ms")
    print(f"  - Postprocessing     : {postprocess_ms:.2f} ms")
    print(f"  - Total Elapsed      : {total_elapsed_ms:.2f} ms")
    if inference_ms > 0:
        print(f"  - Pure Inference FPS : {1000.0 / inference_ms:.1f} FPS")

    print("-" * 70)
    print(f" Detected Objects List ({total_detections} items):")
    if detailed_detections:
        for item in detailed_detections:
            print(
                f"  [{item['id']:02d}] Class: {item['class']:<12} "
                f"| Confidence: {item['confidence'] * 100:>5.1f}% "
                f"| Box [x1, y1, x2, y2]: {item['bbox']}"
            )
    else:
        print("  (No target vehicles found above the confidence threshold.)")

    print("-" * 70)
    print(f" Output Annotated File : {output_filepath.resolve()}")
    print("=" * 70)

    return output_filepath


def main():
    args = parse_arguments()
    print_banner()
    print_educational_note()

    try:
        output_file = run_inference(
            source_path=args.source,
            model_name=args.model,
            output_dir=args.output_dir,
            conf_thresh=args.conf,
            device_arg=args.device,
            all_classes=args.all_classes,
        )
        print(f"\n[SUCCESS] Image inference complete! Output saved to: {output_file}\n")
    except Exception as exc:
        print(f"\n[ERROR] Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
