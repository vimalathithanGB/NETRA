"""
SIH 2026 AI Engine - Dataset Preparation
=========================================
Script: scripts/select_uvh26_subset.py

Purpose:
    Create a class-aware training-image candidate manifest from the locally
    downloaded UVH-26 Majority Voting training annotation file (UVH-26-MV-Train.json).

Context for AI/ML Beginners:
    - Real-world traffic datasets (like UVH-26) often have granular vehicle categories
      (e.g., Hatchback, Sedan, SUV, MUV). For high-performance object detection and
      tracking, we map these into unified operational categories (e.g., 'car').
    - Non-motorized vehicles (Bicycle) or vague labels (Others) are excluded.
    - Instead of randomly selecting images (which might starve minority classes like
      'van' or 'bus'), we first build a 'manifest'—an indexed catalog recording the
      exact classes present in every image. This enables stratified or class-balanced
      sampling later.

Exact Source-to-Target Class Mappings:
    Hatchback       -> car
    Sedan           -> car
    SUV             -> car
    MUV             -> car
    Two-wheeler     -> motorcycle
    Three-wheeler   -> auto_rickshaw
    Bus             -> bus
    Mini-bus        -> bus
    Truck           -> truck
    LCV             -> truck
    Van             -> van
    Tempo-traveller -> van
    Bicycle         -> IGNORED
    Others          -> IGNORED

Target Classes (6 total):
    - car
    - motorcycle
    - auto_rickshaw
    - bus
    - truck
    - van

Output:
    datasets/UVH-26/uvh26_candidate_manifest.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# 1. CLASS MAPPING DEFINITIONS
# ---------------------------------------------------------------------------
# Map granular source classes from UVH-26 into our 6 unified target classes.
SOURCE_TO_TARGET_MAP: Dict[str, str] = {
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Two-wheeler": "motorcycle",
    "Three-wheeler": "auto_rickshaw",
    "Bus": "bus",
    "Mini-bus": "bus",
    "Truck": "truck",
    "LCV": "truck",
    "Van": "van",
    "Tempo-traveller": "van",
}

# Classes explicitly excluded from our vehicle surveillance engine
IGNORED_CLASSES: Set[str] = {"Bicycle", "Others"}

# Ordered list of the 6 target classes for consistent reporting and indexing
TARGET_CLASSES: List[str] = [
    "car",
    "motorcycle",
    "auto_rickshaw",
    "bus",
    "truck",
    "van",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with sensible repository defaults."""
    parser = argparse.ArgumentParser(
        description="Build class-aware candidate manifest from UVH-26 annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="datasets/UVH-26/UVH-26-Train/UVH-26-MV-Train.json",
        help="Path to the UVH-26 COCO-format training JSON annotation file.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="datasets/UVH-26/uvh26_candidate_manifest.csv",
        help="Path to save the generated candidate manifest CSV.",
    )
    return parser.parse_args()


def load_coco_json(json_path: Path) -> dict:
    """
    Load the COCO format JSON file safely.
    Validates file existence and provides a friendly error message if missing.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: '{json_path}'\n"
            f"Please verify that the file exists in the expected path."
        )

    file_size_mb = json_path.stat().st_size / (1024 * 1024)
    print(f"[1/5] Loading COCO annotation JSON ({file_size_mb:.1f} MB): {json_path} ...")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def build_mappings(coco_data: dict) -> Tuple[Dict[int, dict], Dict[int, str]]:
    """
    Build lookup dictionaries for fast O(1) access:
    - image_id -> image metadata dict (file_name, width, height)
    - category_id -> original category_name
    """
    print("[2/5] Building image and category lookup tables ...")

    # Map image_id -> image metadata
    images_map: Dict[int, dict] = {
        img["id"]: img for img in coco_data.get("images", [])
    }

    # Map category_id -> category name string
    categories_map: Dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco_data.get("categories", [])
    }

    return images_map, categories_map


def process_annotations(
    annotations: list,
    categories_map: Dict[int, str],
) -> Tuple[Dict[int, Set[str]], Counter, Counter]:
    """
    Parse each annotation, map source category to target class,
    and aggregate image-level class occurrences and bounding box counts.

    Returns:
        image_target_classes: dict of image_id -> set of target class names present
        target_box_counts: Counter of target_class -> total bounding box count
        ignored_counts: Counter of ignored class name -> count
    """
    print("[3/5] Processing annotations and mapping to 6 target classes ...")

    image_target_classes: Dict[int, Set[str]] = defaultdict(set)
    target_box_counts: Counter = Counter()
    ignored_counts: Counter = Counter()

    for ann in annotations:
        cat_id = ann.get("category_id")
        source_name = categories_map.get(cat_id, "")
        img_id = ann.get("image_id")

        if source_name in IGNORED_CLASSES:
            ignored_counts[source_name] += 1
            continue

        target_class = SOURCE_TO_TARGET_MAP.get(source_name)
        if target_class:
            target_box_counts[target_class] += 1
            image_target_classes[img_id].add(target_class)
        else:
            # Any unmapped category
            ignored_counts[source_name or f"Unknown_ID_{cat_id}"] += 1

    return image_target_classes, target_box_counts, ignored_counts


def generate_manifest(
    images_map: Dict[int, dict],
    image_target_classes: Dict[int, Set[str]],
    output_csv_path: Path,
) -> int:
    """
    Generate candidate manifest CSV containing every image that contains
    at least one of our 6 target classes.

    Columns required:
        - image_id
        - file_name
        - width
        - height
        - source_folder
        - target_classes
        - target_class_count

    Note on source_folder:
        UVH-26 train data resides in partitioned folders:
        UVH-26-Train/data/<folder>/<filename>
        Because the JSON file_name attribute does not include the folder index,
        source_folder is initially recorded as 'UNKNOWN' until images are unpacked.
    """
    print(f"[4/5] Creating candidate manifest: {output_csv_path} ...")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "file_name",
        "width",
        "height",
        "source_folder",
        "target_classes",
        "target_class_count",
    ]

    candidate_count = 0

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over all images in sorted image_id order for deterministic output
        for img_id in sorted(images_map.keys()):
            classes_present = image_target_classes.get(img_id, set())

            # Candidate rule: must contain at least one target class
            if not classes_present:
                continue

            img_meta = images_map[img_id]

            # Sort classes alphabetically for clean, deterministic strings
            sorted_classes = sorted(list(classes_present))
            target_classes_str = ",".join(sorted_classes)
            target_class_count = len(sorted_classes)

            # source_folder is UNKNOWN as specified in requirement 11
            source_folder = "UNKNOWN"

            writer.writerow({
                "image_id": img_id,
                "file_name": img_meta.get("file_name", ""),
                "width": img_meta.get("width", 0),
                "height": img_meta.get("height", 0),
                "source_folder": source_folder,
                "target_classes": target_classes_str,
                "target_class_count": target_class_count,
            })
            candidate_count += 1

    return candidate_count


def print_summary(
    total_images_in_json: int,
    total_annotations_in_json: int,
    candidate_image_count: int,
    target_box_counts: Counter,
    image_target_classes: Dict[int, Set[str]],
    ignored_counts: Counter,
    output_csv_path: Path,
):
    """
    Print an educational, structured terminal summary of dataset statistics.
    """
    print("\n" + "=" * 75)
    print("        UVH-26 CANDIDATE MANIFEST GENERATION - SUMMARY REPORT")
    print("=" * 75)
    print(f" Total Images in JSON           : {total_images_in_json:,}")
    print(f" Total Annotations in JSON      : {total_annotations_in_json:,}")
    print(f" Total Candidate Images (>=1 tgt): {candidate_image_count:,} "
          f"({(candidate_image_count / total_images_in_json * 100):.2f}% of dataset)")
    print(f" Images without Target Classes  : {total_images_in_json - candidate_image_count:,}")

    # Count how many candidate images contain each class
    images_per_class: Counter = Counter()
    for classes in image_target_classes.values():
        for c in classes:
            images_per_class[c] += 1

    total_target_boxes = sum(target_box_counts.values())
    total_ignored_boxes = sum(ignored_counts.values())

    print("-" * 75)
    print(" TARGET CLASS STATISTICS (6 OPERATIONAL CLASSES):")
    print("-" * 75)
    print(f" {'Target Class':<16} | {'Bounding Boxes':<15} | {'Box Share':<10} | {'Images Containing':<18} | {'Image Coverage':<14}")
    print("-" * 75)

    for cls in TARGET_CLASSES:
        boxes = target_box_counts[cls]
        box_pct = (boxes / total_target_boxes * 100.0) if total_target_boxes > 0 else 0.0
        imgs = images_per_class[cls]
        img_pct = (imgs / candidate_image_count * 100.0) if candidate_image_count > 0 else 0.0

        print(
            f" {cls:<16} | {boxes:>10,} boxes | {box_pct:>8.2f}% | "
            f"{imgs:>10,} images | {img_pct:>10.2f}%"
        )

    print("-" * 75)
    print(f" Total Target Bounding Boxes    : {total_target_boxes:,}")
    print(f" Total Ignored Bounding Boxes   : {total_ignored_boxes:,} "
          f"({', '.join([f'{k}: {v:,}' for k, v in ignored_counts.items()])})")

    # Multi-label diversity breakdown
    class_count_dist: Counter = Counter()
    for classes in image_target_classes.values():
        if classes:
            class_count_dist[len(classes)] += 1

    print("-" * 75)
    print(" MULTI-LABEL CO-OCCURRENCE (Diversity per image):")
    for count in sorted(class_count_dist.keys()):
        num_imgs = class_count_dist[count]
        pct = num_imgs / candidate_image_count * 100.0
        print(f"  - Images with exactly {count} target class(es): {num_imgs:>6,} ({pct:>5.1f}%)")

    file_size_kb = output_csv_path.stat().st_size / 1024.0 if output_csv_path.exists() else 0.0
    print("-" * 75)
    print(f" Manifest Saved To             : {output_csv_path.resolve()}")
    print(f" Manifest CSV File Size        : {file_size_kb:.1f} KB")
    print("=" * 75)


def main():
    args = parse_arguments()

    input_json_path = Path(args.input_json)
    output_csv_path = Path(args.output_csv)

    print("=" * 75)
    print("  SIH 2026: UVH-26 TRAINING SUBSET MANIFEST GENERATOR")
    print("=" * 75)

    try:
        # Step 1: Load JSON
        coco_data = load_coco_json(input_json_path)

        total_images = len(coco_data.get("images", []))
        total_annotations = len(coco_data.get("annotations", []))

        # Step 2: Build ID lookup mappings
        images_map, categories_map = build_mappings(coco_data)

        # Step 3: Process annotations and map to target classes
        image_target_classes, target_box_counts, ignored_counts = process_annotations(
            coco_data.get("annotations", []),
            categories_map,
        )

        # Step 4: Generate candidate manifest CSV
        candidate_count = generate_manifest(
            images_map,
            image_target_classes,
            output_csv_path,
        )

        # Step 5: Display statistics and summary
        print_summary(
            total_images_in_json=total_images,
            total_annotations_in_json=total_annotations,
            candidate_image_count=candidate_count,
            target_box_counts=target_box_counts,
            image_target_classes=image_target_classes,
            ignored_counts=ignored_counts,
            output_csv_path=output_csv_path,
        )

        # Mandatory conclusion string specified in user request
        print("[SUCCESS] UVH-26 candidate manifest created.")

    except Exception as exc:
        print(f"\n[ERROR] Failed to generate UVH-26 manifest: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
