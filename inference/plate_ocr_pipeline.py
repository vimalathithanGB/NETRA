"""
NETRA — License Plate Detection & Optical Character Recognition (OCR) Pipeline
File: inference/plate_ocr_pipeline.py
Component: Integration of Model 2 (YOLO Plate Detector) + Model 3 (License Plate OCR)
Project: SIH 2026 NETRA (Networked Engine for Traffic Recognition & Analytics)

Pipeline Flow:
    Input Vehicle Image
          ↓
    YOLOv8n License Plate Detector (best.pt)
          ↓
    Detected Plate Bounding Boxes [x1, y1, x2, y2]
          ↓
    Plate Image Cropping
          ↓
    LicensePlateOCR.predict(crop)
          ↓
    Text Normalization & Indian Plate Disambiguation
          ↓
    Structured JSON Output + Annotated Image
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import our verified LicensePlateOCR module
from ocr.license_plate_ocr import LicensePlateOCR

# Configure module logger
logger = logging.getLogger("NETRA.Pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [NETRA.Pipeline] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default path to trained Model 2 (License Plate Detector)
DEFAULT_PLATE_MODEL = "runs/detect/runs/plate_detection/vehicle_plate_yolov8n_50ep/weights/best.pt"


class PlateOCRPipeline:
    """
    End-to-end integration pipeline combining YOLO license plate detection
    with PaddleOCR recognition-only inference.
    """

    def __init__(
        self,
        weights_path: str = DEFAULT_PLATE_MODEL,
        detector_conf: float = 0.40,
        ocr_conf: float = 0.50,
        device: Optional[str] = None,
        enable_clahe: bool = True,
        enable_fallback: bool = True,
        output_dir: str = "inference/output",
    ) -> None:
        """
        Initializes the detection + OCR pipeline.

        Args:
            weights_path: Path to trained YOLOv8 license plate detector weights.
            detector_conf: Confidence threshold for YOLO plate detection (default: 0.40).
            ocr_conf: Confidence threshold for PaddleOCR text recognition (default: 0.50).
            device: Compute device for YOLO ('cuda:0', 'cpu', or None for auto).
            enable_clahe: Whether to enable CLAHE in OCR preprocessing.
            enable_fallback: Whether to enable OCR fallback on raw crop.
            output_dir: Directory to save annotated visualization images.
        """
        self.weights_path = weights_path
        self.detector_conf = float(detector_conf)
        self.ocr_conf = float(ocr_conf)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Determine compute device for YOLO detector
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"YOLO Detector Device: {self.device}")

        # 2. Load trained YOLO License Plate Detector
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"Trained license plate weights not found at: {self.weights_path}"
            )
        logger.info(f"Loading YOLO License Plate Detector from: {self.weights_path}")
        self.detector = YOLO(self.weights_path)

        # 3. Initialize LicensePlateOCR (always on CPU to protect 4GB VRAM)
        logger.info("Initializing LicensePlateOCR engine on CPU...")
        self.ocr = LicensePlateOCR(
            confidence_threshold=self.ocr_conf,
            enable_clahe=enable_clahe,
            enable_fallback=enable_fallback,
            use_gpu=False,
            lang="en",
        )
        logger.info("PlateOCRPipeline successfully initialized.")

    def predict_image(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Processes a single image through detection and OCR stages.

        Args:
            image_input: Image file path or pre-loaded OpenCV BGR array.

        Returns:
            Dict containing:
                "image_path": str or None
                "image_shape": [H, W, C]
                "num_plates_detected": int
                "plates": List[Dict] with detection & OCR attributes
                "timings_ms": Dict of latency breakdown
        """
        # Load image if file path is provided
        image_path_str: Optional[str] = None
        if isinstance(image_input, (str, Path)):
            image_path_str = str(image_input)
            image = cv2.imread(image_path_str)
            if image is None:
                raise ValueError(f"Failed to read image at: {image_path_str}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        h, w = image.shape[:2]

        # Stage 1: Plate Detection (YOLO)
        t_det_start = time.perf_counter()
        det_results = self.detector.predict(
            source=image,
            conf=self.detector_conf,
            device=self.device,
            verbose=False,
        )
        det_time_ms = (time.perf_counter() - t_det_start) * 1000.0

        # Stage 2: Plate Cropping & OCR Recognition
        plate_records: List[Dict[str, Any]] = []
        total_ocr_time_ms = 0.0

        boxes = det_results[0].boxes if len(det_results) > 0 else []

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box.conf[0].cpu().numpy())

            # Clamp coordinates safely to image boundaries
            x1 = max(0, min(w - 1, int(xyxy[0])))
            y1 = max(0, min(h - 1, int(xyxy[1])))
            x2 = max(x1 + 1, min(w, int(xyxy[2])))
            y2 = max(y1 + 1, min(h, int(xyxy[3])))

            crop = image[y1:y2, x1:x2]

            # Execute OCR on crop
            t_ocr_start = time.perf_counter()
            ocr_res = self.ocr.predict(crop)
            ocr_latency = (time.perf_counter() - t_ocr_start) * 1000.0
            total_ocr_time_ms += ocr_latency

            plate_records.append({
                "bbox": [x1, y1, x2, y2],
                "detector_confidence": round(det_conf, 4),
                "raw_text": ocr_res.get("raw_text", ""),
                "cleaned_text": ocr_res.get("text", ""),
                "ocr_confidence": round(float(ocr_res.get("confidence", 0.0)), 4),
                "valid_format": bool(ocr_res.get("valid_format", False)),
                "ocr_time_ms": round(ocr_latency, 2),
            })

        total_time_ms = det_time_ms + total_ocr_time_ms

        return {
            "image_path": image_path_str,
            "image_shape": [h, w, image.shape[2]],
            "num_plates_detected": len(plate_records),
            "plates": plate_records,
            "timings_ms": {
                "detection": round(det_time_ms, 2),
                "ocr": round(total_ocr_time_ms, 2),
                "total": round(total_time_ms, 2),
            },
        }

    def annotate_image(
        self,
        image: np.ndarray,
        plates: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draws bounding boxes and structured recognition labels on the image.

        Color coding:
        - Vibrant Green (0, 230, 0): Valid Indian License Plate format
        - Vibrant Orange (0, 140, 255): Text detected but non-baseline format / unverified
        - Red (0, 0, 255): No text detected or OCR confidence < threshold
        """
        annotated = image.copy()

        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            det_conf = plate["detector_confidence"]
            text = plate["cleaned_text"]
            ocr_conf = plate["ocr_confidence"]
            is_valid = plate["valid_format"]

            # Select color based on recognition state
            if is_valid:
                box_color = (0, 220, 0)      # Green: Verified plate format
            elif text:
                box_color = (0, 165, 255)    # Orange: Text recognized, format non-standard
            else:
                box_color = (0, 0, 255)      # Red: Unrecognized / low confidence

            # 1. Draw Plate Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # 2. Build Display Label
            display_text = text if text else "PLATE"
            label = f"{display_text} [Det:{det_conf:.2f}|OCR:{ocr_conf:.2f}]"

            # 3. Render Background Badge for High Legibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            badge_x2 = min(annotated.shape[1], x1 + text_w + 10)

            # Draw background rectangle
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (badge_x2, badge_y2),
                box_color,
                -1,
            )

            # Draw white text inside badge
            text_pos = (x1 + 4, badge_y2 - 4)
            cv2.putText(
                annotated,
                label,
                text_pos,
                font,
                font_scale,
                (0, 0, 0) if is_valid or text else (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    def process_and_save(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Executes pipeline on an image file and saves the annotated result image.

        Returns:
            Tuple[Dict, str]: (pipeline_result_dict, saved_annotated_image_path)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        result = self.predict_image(image)
        result["image_path"] = str(image_path)

        # Annotate
        annotated = self.annotate_image(image, result["plates"])

        # Determine output filename
        if output_path is None:
            base_name = Path(image_path).stem
            output_file = self.output_dir / f"{base_name}_plate_ocr.jpg"
        else:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_file), annotated)
        return result, str(output_file)

    def process_batch(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Processes a list of image paths and saves annotated images."""
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        batch_results: List[Dict[str, Any]] = []
        for img_path in image_paths:
            logger.info(f"Processing: {os.path.basename(img_path)}")
            res, out_path = self.process_and_save(img_path)
            res["annotated_image_path"] = out_path
            batch_results.append(res)

        return batch_results


def main() -> None:
    """CLI execution entrypoint for PlateOCRPipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NETRA License Plate Detection & OCR Integration Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image file or directory of images.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_PLATE_MODEL,
        help=f"Path to plate detector weights (default: {DEFAULT_PLATE_MODEL})",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.40,
        help="Confidence threshold for YOLO plate detection (default: 0.40)",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.50,
        help="Confidence threshold for PaddleOCR text recognition (default: 0.50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference/output/plate_ocr",
        help="Directory where annotated output images are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for YOLO ('cuda:0', 'cpu', or auto).",
    )

    args = parser.parse_args()

    pipeline = PlateOCRPipeline(
        weights_path=args.weights,
        detector_conf=args.detector_conf,
        ocr_conf=args.ocr_conf,
        device=args.device,
        output_dir=args.output_dir,
    )

    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [str(source_path)]
    elif source_path.is_dir():
        image_files = [
            str(p) for p in source_path.glob("*.jpg")
        ] + [str(p) for p in source_path.glob("*.png")]
    else:
        print(f"ERROR: Source path does not exist: {args.source}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to process.")
    for idx, img_file in enumerate(image_files, 1):
        res, out_img = pipeline.process_and_save(img_file)
        print("=" * 60)
        print(f"[{idx}/{len(image_files)}] Image: {os.path.basename(img_file)}")
        print(f"Resolution: {res['image_shape'][1]}x{res['image_shape'][0]}")
        print(f"Plates Detected: {res['num_plates_detected']}")
        print(f"Latency: Detection={res['timings_ms']['detection']}ms | OCR={res['timings_ms']['ocr']}ms | Total={res['timings_ms']['total']}ms")
        for p_idx, plate in enumerate(res["plates"], 1):
            print(f"  Plate {p_idx}:")
            print(f"    BBox:       {plate['bbox']}")
            print(f"    Det Conf:   {plate['detector_confidence']:.4f}")
            print(f"    Raw Text:   {plate['raw_text']}")
            print(f"    Clean Text: {plate['cleaned_text']}")
            print(f"    OCR Conf:   {plate['ocr_confidence']:.4f}")
            print(f"    Valid:      {plate['valid_format']}")
        print(f"Annotated Image Saved: {out_img}")
        print("=" * 60)


if __name__ == "__main__":
    main()
