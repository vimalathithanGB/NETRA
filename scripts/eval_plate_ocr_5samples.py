"""
NETRA — 5-Image Plate Detection + OCR Integration Evaluation
Script: scripts/eval_plate_ocr_5samples.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.plate_ocr_pipeline import PlateOCRPipeline

def main():
    pipeline = PlateOCRPipeline(
        weights_path="runs/detect/runs/plate_detection/vehicle_plate_yolov8n_50ep/weights/best.pt",
        detector_conf=0.40,
        ocr_conf=0.50,
        output_dir="inference/output/plate_ocr"
    )

    test_files = [
        "datasets/indian_license_plate_clean/test/images/20220630_19_32_53_371_000_NICf9pZpG7Wd4tmRH1tzGvEfPNn1_F_3000_4000_jpg.rf.8dc6fe5d88c8f95d917ea5ac227550ac.jpg",
        "datasets/indian_license_plate_clean/test/images/165eaa92-8e90-44e6-b132-8af462bae0c9___scorpio-rear_JPG_jpeg.rf.a07ae1d79d0472419bceca67606923f3.jpg",
        "datasets/indian_license_plate_clean/test/images/car-wbs-KL10AW2814_00000_jpeg.rf.b49980ccc823c8b9378dfde197204a8d.jpg",
        "datasets/indian_license_plate_clean/test/images/car-wbs-MH06AW8929_00001_jpeg.rf.0bf557e9854d5ef9de9659956a351291.jpg",
        "datasets/indian_license_plate_clean/test/images/20220714_09_14_20_539_000_ZaG8yzjrIdPimkMRBMHzy2ylhP62_T_5120_3840_jpg.rf.3119c2d942afb9a4a4b68df7e1aded09.jpg",
    ]

    results = pipeline.process_batch(test_files)

    print("\n" + "=" * 70)
    print("NETRA 5-IMAGE PLATE DETECTION + OCR INTEGRATION EVALUATION")
    print("=" * 70)

    for i, res in enumerate(results, 1):
        img_name = os.path.basename(res["image_path"])
        print(f"\n--- [Image {i}/5]: {img_name} ---")
        print(f"Resolution:        {res['image_shape'][1]}x{res['image_shape'][0]}")
        print(f"Plates Detected:   {res['num_plates_detected']}")
        print(f"Detection Latency: {res['timings_ms']['detection']:.2f} ms")
        print(f"OCR Latency:       {res['timings_ms']['ocr']:.2f} ms")
        print(f"Total Latency:     {res['timings_ms']['total']:.2f} ms")
        print(f"Annotated Image:   {res['annotated_image_path']}")

        for p_idx, p in enumerate(res["plates"], 1):
            print(f"  * Plate {p_idx}:")
            print(f"      Bounding Box:        {p['bbox']}")
            print(f"      Detector Confidence: {p['detector_confidence']:.4f}")
            print(f"      Raw OCR Text:        '{p['raw_text']}'")
            print(f"      Cleaned Text:        '{p['cleaned_text']}'")
            print(f"      OCR Confidence:      {p['ocr_confidence']:.4f}")
            print(f"      Format Valid:        {p['valid_format']}")

    # Save complete JSON summary
    summary_path = "inference/output/plate_ocr/5samples_evaluation.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured evaluation JSON to: {summary_path}")

if __name__ == "__main__":
    main()
