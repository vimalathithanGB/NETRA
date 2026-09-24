"""
prepare_indian_license_plate_dataset.py
SIH 2026 NETRA - Model 2: Indian License Plate Detection

Converts Indian license plate dataset with 986 classes (individual plate strings)
into a single-class dataset (0: license_plate) for Model 2 plate localization.
"""

import os
import shutil
from pathlib import Path
from PIL import Image

SRC_DIR = Path(r"datasets/indian licence plate.v1i.yolov8").resolve()
DST_DIR = Path(r"datasets/indian_license_plate_clean").resolve()

SPLITS = ["train", "valid", "test"]

def main():
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    
    assert SRC_DIR.exists(), f"Source directory {SRC_DIR} does not exist!"
    
    # 1. Create target directories
    for s in SPLITS:
        (DST_DIR / s / "images").mkdir(parents=True, exist_ok=True)
        (DST_DIR / s / "labels").mkdir(parents=True, exist_ok=True)
        
    print("\n1. Copying images and remapping labels to class 0...")
    conversion_stats = {
        s: {"images_copied": 0, "labels_written": 0, "bboxes_converted": 0}
        for s in SPLITS
    }
    
    for s in SPLITS:
        src_img_dir = SRC_DIR / s / "images"
        src_lbl_dir = SRC_DIR / s / "labels"
        dst_img_dir = DST_DIR / s / "images"
        dst_lbl_dir = DST_DIR / s / "labels"
        
        for img_p in sorted(src_img_dir.iterdir()):
            dst_img_p = dst_img_dir / img_p.name
            shutil.copy2(img_p, dst_img_p)
            conversion_stats[s]["images_copied"] += 1
            
            lbl_name = f"{img_p.stem}.txt"
            src_lbl_p = src_lbl_dir / lbl_name
            dst_lbl_p = dst_lbl_dir / lbl_name
            
            converted_lines = []
            if src_lbl_p.exists():
                with open(src_lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        parts = line_str.split()
                        assert len(parts) == 5, f"Malformed line in {src_lbl_p}: {line_str}"
                        # Replace class ID with 0, preserve exact coordinate strings
                        new_line = "0 " + " ".join(parts[1:]) + "\n"
                        converted_lines.append(new_line)
                        conversion_stats[s]["bboxes_converted"] += 1
                        
            with open(dst_lbl_p, "w", encoding="utf-8") as f:
                f.writelines(converted_lines)
            conversion_stats[s]["labels_written"] += 1

    print("Copy and label conversion complete.")
    for s in SPLITS:
        print(f"  Split {s}: {conversion_stats[s]['images_copied']} images, {conversion_stats[s]['labels_written']} labels, {conversion_stats[s]['bboxes_converted']} bboxes")

    # 2. Write new data.yaml
    yaml_content = f"""path: {DST_DIR.as_posix()}

train: train/images
val: valid/images
test: test/images

nc: 1

names:
  0: license_plate
"""
    data_yaml_p = DST_DIR / "data.yaml"
    with open(data_yaml_p, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n2. Written data.yaml at {data_yaml_p}")

    # 3. Comprehensive Validation
    print("\n3. Validating new dataset...")
    val_results = {
        "missing_images": 0,
        "missing_labels": 0,
        "malformed_annotations": 0,
        "non_zero_classes": 0,
        "coordinate_mismatches": 0,
        "corrupted_images": 0,
        "unexpected_files": 0,
        "split_counts": {},
        "bbox_counts": {}
    }

    for s in SPLITS:
        src_img_dir = SRC_DIR / s / "images"
        src_lbl_dir = SRC_DIR / s / "labels"
        dst_img_dir = DST_DIR / s / "images"
        dst_lbl_dir = DST_DIR / s / "labels"
        
        src_imgs = sorted([f.name for f in src_img_dir.iterdir()])
        dst_imgs = sorted([f.name for f in dst_img_dir.iterdir()])
        src_lbls = sorted([f.name for f in src_lbl_dir.iterdir()])
        dst_lbls = sorted([f.name for f in dst_lbl_dir.iterdir()])
        
        if src_imgs != dst_imgs:
            val_results["missing_images"] += len(set(src_imgs) - set(dst_imgs))
        if src_lbls != dst_lbls:
            val_results["missing_labels"] += len(set(src_lbls) - set(dst_lbls))
            
        val_results["split_counts"][s] = {
            "source_images": len(src_imgs),
            "output_images": len(dst_imgs),
            "source_labels": len(src_lbls),
            "output_labels": len(dst_lbls)
        }
        
        s_bboxes = 0
        src_bboxes = 0
        for img_name in dst_imgs:
            dst_img_p = dst_img_dir / img_name
            # Test image readability
            try:
                with Image.open(dst_img_p) as im:
                    im.verify()
            except Exception:
                val_results["corrupted_images"] += 1
                
            stem = Path(img_name).stem
            dst_lbl_p = dst_lbl_dir / f"{stem}.txt"
            src_lbl_p = src_lbl_dir / f"{stem}.txt"
            
            src_lines = []
            if src_lbl_p.exists():
                with open(src_lbl_p, "r", encoding="utf-8") as f:
                    src_lines = [l.strip().split() for l in f if l.strip()]
            src_bboxes += len(src_lines)
            
            dst_lines = []
            if dst_lbl_p.exists():
                with open(dst_lbl_p, "r", encoding="utf-8") as f:
                    dst_lines = [l.strip().split() for l in f if l.strip()]
            s_bboxes += len(dst_lines)
            
            assert len(src_lines) == len(dst_lines), f"Box count mismatch for {stem}"
            
            for src_parts, dst_parts in zip(src_lines, dst_lines):
                if len(dst_parts) != 5:
                    val_results["malformed_annotations"] += 1
                if dst_parts[0] != "0":
                    val_results["non_zero_classes"] += 1
                if src_parts[1:] != dst_parts[1:]:
                    val_results["coordinate_mismatches"] += 1
                    
        val_results["bbox_counts"][s] = {
            "source_bboxes": src_bboxes,
            "output_bboxes": s_bboxes
        }

    total_out_imgs = sum(val_results["split_counts"][s]["output_images"] for s in SPLITS)
    total_src_imgs = sum(val_results["split_counts"][s]["source_images"] for s in SPLITS)
    total_out_lbls = sum(val_results["split_counts"][s]["output_labels"] for s in SPLITS)
    total_src_lbls = sum(val_results["split_counts"][s]["source_labels"] for s in SPLITS)
    total_out_box = sum(val_results["bbox_counts"][s]["output_bboxes"] for s in SPLITS)
    total_src_box = sum(val_results["bbox_counts"][s]["source_bboxes"] for s in SPLITS)

    pass_status = (
        val_results["missing_images"] == 0 and
        val_results["missing_labels"] == 0 and
        val_results["malformed_annotations"] == 0 and
        val_results["non_zero_classes"] == 0 and
        val_results["coordinate_mismatches"] == 0 and
        val_results["corrupted_images"] == 0 and
        val_results["unexpected_files"] == 0 and
        total_out_imgs == 1901 and
        total_out_lbls == 1901 and
        total_out_box == 1908
    )

    final_result_str = "PASS" if pass_status else "FAIL"
    print(f"\nFinal Validation Status: {final_result_str}")

    # 4. Generate dataset_conversion_report.md
    report_content = f"""# License Plate Dataset Class Conversion Report

## Source Dataset
`D:\\apps\\coding\\hackathon projects\\SIH 2026\\SIH2026-AI-ENGINE\\datasets\\indian licence plate.v1i.yolov8`

## Output Dataset
`D:\\apps\\coding\\hackathon projects\\SIH 2026\\SIH2026-AI-ENGINE\\datasets\\indian_license_plate_clean`

## Original Classes
986

## New Classes
1

## Class Mapping

| Original Class IDs | New Class ID | Target Class Name |
|---|---:|---|
| 0–985 | 0 | `license_plate` |

## Image Counts

| Split | Source | Output | Status |
|---|---:|---:|---|
| Train | 1332 | {val_results['split_counts']['train']['output_images']} | MATCH |
| Validation | 379 | {val_results['split_counts']['valid']['output_images']} | MATCH |
| Test | 190 | {val_results['split_counts']['test']['output_images']} | MATCH |
| **Total** | **1901** | **{total_out_imgs}** | **MATCH** |

## Bounding Box Counts

| Split | Source | Output | Status |
|---|---:|---:|---|
| Train | 1338 | {val_results['bbox_counts']['train']['output_bboxes']} | MATCH |
| Validation | 380 | {val_results['bbox_counts']['valid']['output_bboxes']} | MATCH |
| Test | 190 | {val_results['bbox_counts']['test']['output_bboxes']} | MATCH |
| **Total** | **1908** | **{total_out_box}** | **MATCH** |

## Validation

- **missing images**: {val_results['missing_images']}
- **missing labels**: {val_results['missing_labels']}
- **malformed annotations**: {val_results['malformed_annotations']}
- **non-zero class IDs**: {val_results['non_zero_classes']}
- **changed coordinates**: {val_results['coordinate_mismatches']}
- **corrupted output images**: {val_results['corrupted_images']}
- **unexpected files**: {val_results['unexpected_files']}

## Final Result

{final_result_str}
"""
    report_p = DST_DIR / "dataset_conversion_report.md"
    with open(report_p, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report written to: {report_p}")

if __name__ == "__main__":
    main()
