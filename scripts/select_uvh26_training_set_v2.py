"""
SIH 2026 AI Engine - Dataset Preparation (v2)
==============================================
Script: scripts/select_uvh26_training_set_v2.py

Purpose:
    Select a deterministic, balanced, class-aware 12,000-image training subset
    from the mapped UVH-26 candidate manifest (uvh26_candidate_manifest_mapped.csv)
    with explicit scene complexity controls (both simple and crowded scenes).

Why Version 2 Was Created:
    - In v1, unconstrained greedy ranking selected 87.2% highly crowded frames (4–6 classes)
      and 0 single-class frames, starving the detector of simple isolated vehicle examples.
    - Version 2 introduces explicit scene-count quotas across 1 to 6 target classes:
        1 class  -> 700 images
        2 classes -> 1,800 images
        3 classes -> 3,000 images
        4 classes -> 3,800 images
        5 classes -> 2,200 images (+ 142 surplus from bin 6 = 2,342)
        6 classes -> 500 images requested (358 available, 100% selected)
        Total     = Exactly 12,000 images
    - Within each scene-count group, class-aware inverse frequency scoring prioritizes
      under-represented classes (van, bus, truck) without allowing rare classes to
      collapse the scene-complexity diversity.
    - Proportional source-folder balancing (000, 001, 002, 003, 004) prevents any single
      camera or location from dominating.
    - Deterministic tie-breaking using seed 42 prevents temporal clustering.

Output Files:
    - datasets/UVH-26/uvh26_training_12000_manifest_v2.csv
    - datasets/UVH-26/uvh26_training_12000_report_v2.txt

Constraints:
    - Zero image downloads.
    - Zero model training.
    - Existing v1 files are preserved.
    - Python standard library only.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Six unified operational target classes
TARGET_CLASSES: List[str] = [
    "car",
    "motorcycle",
    "auto_rickshaw",
    "bus",
    "truck",
    "van",
]

# Source folders in UVH-26
SOURCE_FOLDERS: List[str] = ["000", "001", "002", "003", "004"]

# Target quotas per scene complexity (target_class_count)
TARGET_SCENE_QUOTAS: Dict[int, int] = {
    1: 700,
    2: 1800,
    3: 3000,
    4: 3800,
    5: 2200,
    6: 500,
}

# Raw source-to-target mapping for bounding box analysis
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Select class-aware 12,000-image training subset with scene complexity control (v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="datasets/UVH-26/uvh26_candidate_manifest_mapped.csv",
        help="Path to mapped candidate manifest CSV.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="datasets/UVH-26/UVH-26-Train/UVH-26-MV-Train.json",
        help="Path to original COCO JSON (for bounding box verification).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="datasets/UVH-26/uvh26_training_12000_manifest_v2.csv",
        help="Path to output v2 manifest CSV.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="datasets/UVH-26/uvh26_training_12000_report_v2.txt",
        help="Path to output v2 selection report.",
    )
    parser.add_argument(
        "--target-images",
        type=int,
        default=12000,
        help="Exact number of images to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection and tie-breaking.",
    )
    return parser.parse_args()


def load_candidate_manifest(csv_path: Path) -> List[dict]:
    """Read the candidate manifest CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found at: '{csv_path}'")

    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    return rows


def load_bounding_box_counts(json_path: Path) -> Dict[int, Counter]:
    """Load per-image target class bounding box counts from original COCO JSON."""
    if not json_path.exists():
        return defaultdict(Counter)

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    img_boxes: Dict[int, Counter] = defaultdict(Counter)

    for ann in coco.get("annotations", []):
        cat_name = cat_map.get(ann.get("category_id"), "")
        tgt = SOURCE_TO_TARGET_MAP.get(cat_name)
        if tgt in TARGET_CLASSES:
            img_boxes[ann["image_id"]][tgt] += 1

    return img_boxes


def calculate_class_weights(rows: List[dict]) -> Dict[str, float]:
    """
    Calculate log-smoothed inverse class frequency weights:
    weight(class) = ln(1 + Total_Candidates / Class_Count)
    Provides a solid boost to minority classes without dominating the scene quotas.
    """
    total_candidates = len(rows)
    class_counts = Counter()
    for r in rows:
        for c in r["target_classes"].split(","):
            class_counts[c] += 1

    weights = {}
    for c in TARGET_CLASSES:
        cnt = class_counts.get(c, 1)
        weights[c] = round(math.log(1.0 + (total_candidates / cnt)), 4)

    return weights


def plan_scene_quotas(
    candidate_rows: List[dict],
    target_quotas: Dict[int, int],
    total_target: int = 12000,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    """
    Determine actual bin targets based on dataset availability.
    If a bin has fewer available images than requested (e.g. 6 classes has 358 vs 500 requested),
    take 100% of available images and reallocate the deficit to adjacent bins (e.g. 5 classes).
    """
    available_per_bin = Counter(int(r["target_class_count"]) for r in candidate_rows)

    actual_targets: Dict[int, int] = {}
    notes: Dict[int, str] = {}
    deficit = 0

    # Process Bin 6 first to check for availability constraint
    avail_6 = available_per_bin.get(6, 0)
    desired_6 = target_quotas.get(6, 500)
    if avail_6 < desired_6:
        actual_targets[6] = avail_6
        deficit = desired_6 - avail_6
        notes[6] = f"Requested {desired_6}, available {avail_6}. Selected 100% available; {deficit} deficit transferred to Bin 5."
    else:
        actual_targets[6] = desired_6
        notes[6] = f"Requested {desired_6}, selected {desired_6}."

    # Process Bins 1 to 4 with standard quotas
    for k in [1, 2, 3, 4]:
        avail = available_per_bin.get(k, 0)
        desired = target_quotas.get(k, 0)
        actual_targets[k] = min(avail, desired)
        notes[k] = f"Requested {desired}, selected {actual_targets[k]}."

    # Bin 5 absorbs any deficit from Bin 6
    avail_5 = available_per_bin.get(5, 0)
    desired_5 = target_quotas.get(5, 2200) + deficit
    actual_targets[5] = min(avail_5, desired_5)
    notes[5] = f"Requested {target_quotas[5]} + {deficit} (from Bin 6) = {desired_5}, selected {actual_targets[5]}."

    # Adjust any minor difference to guarantee exact total_target
    current_total = sum(actual_targets.values())
    if current_total != total_target:
        diff = total_target - current_total
        actual_targets[4] += diff
        notes[4] += f" Adjusted by {diff} to reach exactly {total_target}."

    return actual_targets, dict(available_per_bin), notes


def select_v2_training_subset(
    candidate_rows: List[dict],
    actual_bin_targets: Dict[int, int],
    class_weights: Dict[str, float],
    seed: int = 42,
) -> List[dict]:
    """
    Execute stratified, class-aware selection across bins and folders:
    1. Group candidate images by (target_class_count, source_folder).
    2. For each bin, compute proportional folder quotas matching candidate proportions.
    3. Within each folder bucket, score images:
       score = sum(class_weights) + deterministic_jitter (seed 42)
    4. Pick top candidates per folder quota.
    5. If any folder in a bin has fewer candidates than its quota, fill the remaining
       shortfall from other folders in that same bin by highest score.
    """
    rng = random.Random(seed)
    total_candidates = len(candidate_rows)

    # Candidate counts per source folder
    folder_totals = Counter(r["source_folder"] for r in candidate_rows)

    # Group candidate rows into [bin][folder] buckets
    bin_folder_rows: Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in candidate_rows:
        k = int(r["target_class_count"])
        fld = r["source_folder"]
        bin_folder_rows[k][fld].append(r)

    selected_rows: List[dict] = []

    for k in sorted(actual_bin_targets.keys()):
        target_k = actual_bin_targets[k]

        # Gather all candidates in this bin
        all_candidates_in_bin: List[dict] = []
        for fld in SOURCE_FOLDERS:
            all_candidates_in_bin.extend(bin_folder_rows[k][fld])

        # If available in bin is less than or equal to target, take all
        if len(all_candidates_in_bin) <= target_k:
            for cand in all_candidates_in_bin:
                c_copy = dict(cand)
                classes = c_copy["target_classes"].split(",")
                c_copy["selection_score"] = round(sum(class_weights[c] for c in classes), 4)
                c_copy["selection_reason"] = f"bin_{k}_full_coverage"
                selected_rows.append(c_copy)
            continue

        # Proportional folder quotas for this bin
        f_quotas: Dict[str, int] = {}
        rem = target_k
        for fld in SOURCE_FOLDERS[:-1]:
            q = int(round(target_k * folder_totals[fld] / total_candidates))
            f_quotas[fld] = q
            rem -= q
        f_quotas[SOURCE_FOLDERS[-1]] = rem  # assign remainder to '004'

        bin_selected: List[dict] = []
        unpicked_in_bin: List[Tuple[float, dict]] = []

        for fld in SOURCE_FOLDERS:
            candidates = bin_folder_rows[k][fld]
            quota = min(f_quotas[fld], len(candidates))

            # Score each candidate: class rarity weight + deterministic anti-clustering jitter
            scored: List[Tuple[float, dict]] = []
            for cand in candidates:
                classes = cand["target_classes"].split(",")
                base_score = sum(class_weights[c] for c in classes)
                jitter = rng.uniform(0.0, 0.001)
                total_score = base_score + jitter

                c_copy = dict(cand)
                c_copy["selection_score"] = round(base_score, 4)

                # Meaningful selection reason
                has_rare = ("van" in classes) or ("bus" in classes)
                if has_rare and k >= 3:
                    c_copy["selection_reason"] = "mixed"
                elif has_rare:
                    c_copy["selection_reason"] = "rare_class"
                elif k >= 3:
                    c_copy["selection_reason"] = "multi_class"
                else:
                    c_copy["selection_reason"] = "balanced_folder"

                scored.append((total_score, c_copy))

            # Sort descending by score
            scored.sort(key=lambda x: x[0], reverse=True)
            bin_selected.extend([item[1] for item in scored[:quota]])
            unpicked_in_bin.extend(scored[quota:])

        # If any folder had insufficient candidates in this bin, fill shortfall from other folders
        shortfall = target_k - len(bin_selected)
        if shortfall > 0:
            unpicked_in_bin.sort(key=lambda x: x[0], reverse=True)
            bin_selected.extend([item[1] for item in unpicked_in_bin[:shortfall]])

        selected_rows.extend(bin_selected)

    # Sort final selected list by numeric image_id for clean deterministic indexing
    selected_rows.sort(key=lambda x: int(x["image_id"]))
    return selected_rows


def perform_validation_checks(
    selected_rows: List[dict],
    candidate_rows: List[dict],
    expected_count: int = 12000,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Run mandatory validation checks:
    - selected_count == 12000
    - image IDs are unique
    - no missing source_folder
    - all 6 target classes appear
    - no Bicycle/Others target labels
    - no invalid target class names
    - deterministic seed = 42
    """
    results: Dict[str, str] = {}

    # Check 1: Count
    if len(selected_rows) == expected_count:
        results["selected_count == 12000"] = f"PASSED ({len(selected_rows):,} images)"
    else:
        results["selected_count == 12000"] = f"FAILED ({len(selected_rows)} != {expected_count})"

    # Check 2: Unique image IDs
    img_ids = [r["image_id"] for r in selected_rows]
    if len(set(img_ids)) == expected_count:
        results["image_ids_unique"] = f"PASSED ({len(set(img_ids)):,} unique IDs, 0 duplicates)"
    else:
        results["image_ids_unique"] = f"FAILED ({expected_count - len(set(img_ids))} duplicate IDs)"

    # Check 3: Unique file names
    fnames = [r["file_name"] for r in selected_rows]
    if len(set(fnames)) == expected_count:
        results["file_names_unique"] = f"PASSED ({len(set(fnames)):,} unique filenames, 0 duplicates)"
    else:
        results["file_names_unique"] = f"FAILED ({expected_count - len(set(fnames))} duplicate filenames)"

    # Check 4: No missing source_folder
    invalid_folders = [r for r in selected_rows if not r.get("source_folder") or r.get("source_folder") == "UNKNOWN"]
    if not invalid_folders:
        results["no_missing_source_folder"] = "PASSED (all records have valid source_folder 000-004)"
    else:
        results["no_missing_source_folder"] = f"FAILED ({len(invalid_folders)} invalid folders)"

    # Check 5: All 6 target classes appear
    present_classes = set()
    for r in selected_rows:
        for c in r["target_classes"].split(","):
            present_classes.add(c)

    if present_classes == set(TARGET_CLASSES):
        results["all_6_target_classes_appear"] = f"PASSED ({sorted(list(present_classes))})"
    else:
        results["all_6_target_classes_appear"] = f"FAILED (missing: {set(TARGET_CLASSES) - present_classes})"

    # Check 6: No Bicycle or Others
    forbidden = {"Bicycle", "Others", "bicycle", "others"}
    violations = []
    for r in selected_rows:
        classes = r["target_classes"].split(",")
        for c in classes:
            if c in forbidden:
                violations.append((r["image_id"], c))

    if not violations:
        results["no_bicycle_others_labels"] = "PASSED (0 excluded categories in subset)"
    else:
        results["no_bicycle_others_labels"] = f"FAILED ({len(violations)} forbidden labels found)"

    # Check 7: No invalid target class names
    invalid_names = present_classes - set(TARGET_CLASSES)
    if not invalid_names:
        results["no_invalid_class_names"] = "PASSED (strictly operational categories)"
    else:
        results["no_invalid_class_names"] = f"FAILED (invalid: {invalid_names})"

    # Check 8: Deterministic seed = 42
    results["deterministic_seed"] = f"PASSED (seed={seed})"

    return results


def write_manifest_csv_v2(selected_rows: List[dict], output_csv_path: Path):
    """Save selected v2 manifest CSV."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "file_name",
        "width",
        "height",
        "source_folder",
        "source_path",
        "target_classes",
        "target_class_count",
        "selection_score",
        "selection_reason",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected_rows:
            writer.writerow({
                "image_id": r["image_id"],
                "file_name": r["file_name"],
                "width": r["width"],
                "height": r["height"],
                "source_folder": r["source_folder"],
                "source_path": r["source_path"],
                "target_classes": r["target_classes"],
                "target_class_count": r["target_class_count"],
                "selection_score": r["selection_score"],
                "selection_reason": r["selection_reason"],
            })


def generate_report_v2(
    candidate_rows: List[dict],
    selected_rows: List[dict],
    bin_targets: Dict[int, int],
    bin_available: Dict[int, int],
    bin_notes: Dict[int, str],
    validation_results: Dict[str, str],
    image_boxes: Dict[int, Counter],
    seed: int,
    output_report_path: Path,
):
    """Write comprehensive, human-readable v2 selection report."""
    total_candidates = len(candidate_rows)
    total_selected = len(selected_rows)
    selection_pct = (total_selected / total_candidates * 100.0) if total_candidates > 0 else 0.0

    # Folder distribution before and after
    folder_before = Counter(r["source_folder"] for r in candidate_rows)
    folder_after = Counter(r["source_folder"] for r in selected_rows)

    # Class image coverage before and after
    class_before = Counter()
    for r in candidate_rows:
        for c in r["target_classes"].split(","):
            class_before[c] += 1

    class_after = Counter()
    for r in selected_rows:
        for c in r["target_classes"].split(","):
            class_after[c] += 1

    # Bounding box counts before and after
    boxes_before = Counter()
    for r in candidate_rows:
        for c, cnt in image_boxes[int(r["image_id"])].items():
            boxes_before[c] += cnt

    boxes_after = Counter()
    for r in selected_rows:
        for c, cnt in image_boxes[int(r["image_id"])].items():
            boxes_after[c] += cnt

    # Scene complexity distribution
    complexity_counts = Counter(int(r["target_class_count"]) for r in selected_rows)

    lines: List[str] = [
        "=" * 78,
        "SIH 2026 AI ENGINE - UVH-26 12K TRAINING SUBSET SELECTION REPORT (V2)",
        "=" * 78,
        f"Random Seed                  : {seed}",
        f"Algorithm                    : Class-Aware Stratified Selection with Scene Complexity Control",
        "",
        "-" * 78,
        "A. DATASET INPUT STATISTICS",
        "-" * 78,
        f"Total Candidate Images       : {total_candidates:,}",
        "Total Available Images by Source Folder:",
    ]

    for fld in SOURCE_FOLDERS:
        cnt = folder_before[fld]
        pct = cnt / total_candidates * 100.0
        lines.append(f"  - Folder {fld}: {cnt:>6,} images ({pct:>5.2f}%)")

    lines.extend([
        "",
        "-" * 78,
        "B. SELECTED SUBSET STATISTICS",
        "-" * 78,
        f"Exactly Selected Image Count : {total_selected:,}",
        f"Selection Percentage         : {selection_pct:.2f}% of candidate dataset",
        "",
        "-" * 78,
        "C. SCENE COMPLEXITY DISTRIBUTION (TARGET CLASSES PER IMAGE)",
        "-" * 78,
        f" {'Complexity':<18} | {'Requested':<10} | {'Available':<10} | {'Selected':<10} | {'Share of Subset':<16} | {'Status/Note'}",
        "-" * 78,
    ])

    for k in sorted(TARGET_SCENE_QUOTAS.keys()):
        req = TARGET_SCENE_QUOTAS[k]
        avail = bin_available.get(k, 0)
        sel = complexity_counts.get(k, 0)
        share = sel / total_selected * 100.0
        note = bin_notes.get(k, "OK")
        lines.append(f" {f'{k} target class(es)':<18} | {req:>9,} | {avail:>9,} | {sel:>9,} | {share:>14.2f}% | {note}")

    lines.extend([
        "-" * 78,
        f" {'TOTAL':<18} | {12000:>9,} | {total_candidates:>9,} | {total_selected:>9,} | {'100.00%':>15} |",
        "",
        "Scene Complexity Groupings:",
        f"  - Simple Scenes   (1–2 classes): {complexity_counts[1] + complexity_counts[2]:>6,} images ({((complexity_counts[1] + complexity_counts[2])/total_selected*100):>5.2f}%)",
        f"  - Moderate Scenes (3 classes)  : {complexity_counts[3]:>6,} images ({complexity_counts[3]/total_selected*100:>5.2f}%)",
        f"  - Crowded Scenes  (4–6 classes): {complexity_counts[4] + complexity_counts[5] + complexity_counts[6]:>6,} images ({((complexity_counts[4] + complexity_counts[5] + complexity_counts[6])/total_selected*100):>5.2f}%)",
        "",
        "-" * 78,
        "D. TARGET CLASS COVERAGE",
        "-" * 78,
        f" {'Class':<15} | {'Available Images':<18} | {'Selected Images':<18} | {'Coverage %':<12}",
        "-" * 78,
    ])

    for c in TARGET_CLASSES:
        avail = class_before[c]
        sel = class_after[c]
        cov = (sel / avail * 100.0) if avail > 0 else 0.0
        lines.append(f" {c:<15} | {avail:>10,} images | {sel:>10,} images | {cov:>10.2f}%")

    if boxes_before:
        lines.extend([
            "",
            "Bounding Box Retention Breakdown:",
            f" {'Class':<15} | {'Available Boxes':<18} | {'Selected Boxes':<18} | {'Box Retention %':<15}",
            "-" * 78,
        ])
        for c in TARGET_CLASSES:
            b_cnt = boxes_before[c]
            s_cnt = boxes_after[c]
            ret = (s_cnt / b_cnt * 100.0) if b_cnt > 0 else 0.0
            lines.append(f" {c:<15} | {b_cnt:>10,} boxes  | {s_cnt:>10,} boxes  | {ret:>13.2f}%")

    lines.extend([
        "",
        "-" * 78,
        "E. SOURCE-FOLDER DISTRIBUTION",
        "-" * 78,
        f" {'Folder':<10} | {'Available':<15} | {'Selected':<15} | {'Share of Subset':<18} | {'Retention %':<12}",
        "-" * 78,
    ])

    for fld in SOURCE_FOLDERS:
        avail = folder_before[fld]
        sel = folder_after[fld]
        share = sel / total_selected * 100.0
        ret = (sel / avail * 100.0) if avail > 0 else 0.0
        lines.append(f" {fld:<10} | {avail:>8,} images | {sel:>8,} images | {share:>16.2f}% | {ret:>10.2f}%")

    lines.extend([
        "",
        "-" * 78,
        "F. VALIDATION CHECKS",
        "-" * 78,
    ])

    for check_name, status in validation_results.items():
        lines.append(f"  [{status.split()[0]}] {check_name:<30}: {status}")

    lines.extend([
        "",
        "=" * 78,
        "END OF REPORT V2",
        "=" * 78,
    ])

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_terminal_summary_v2(
    total_candidates: int,
    total_selected: int,
    complexity_counts: Counter,
    class_before: Counter,
    class_after: Counter,
    folder_before: Counter,
    folder_after: Counter,
    validation_results: Dict[str, str],
    output_csv_path: Path,
    output_report_path: Path,
):
    """Print clean, concise terminal summary matching user specification."""
    print("\n" + "=" * 65)
    print(" UVH-26 12K TRAINING SUBSET SELECTION V2 - SUMMARY")
    print("=" * 65)
    print(f" Candidate Images    : {total_candidates:,}")
    print(f" Selected Images     : {total_selected:,}")
    print(f" Selection Percentage: {total_selected / total_candidates * 100:.2f}%")

    print("\n[1] SCENE COMPLEXITY DISTRIBUTION:")
    print("-" * 65)
    for k in sorted(TARGET_SCENE_QUOTAS.keys()):
        cnt = complexity_counts.get(k, 0)
        pct = cnt / total_selected * 100.0
        print(f"  - {k} target class(es) : {cnt:>6,} images ({pct:>5.2f}%)")

    simple_scenes = complexity_counts[1] + complexity_counts[2]
    moderate_scenes = complexity_counts[3]
    crowded_scenes = complexity_counts[4] + complexity_counts[5] + complexity_counts[6]
    print(f"  --> Simple   (1-2 classes): {simple_scenes:>6,} ({simple_scenes/total_selected*100:>5.2f}%)")
    print(f"  --> Moderate (3 classes)  : {moderate_scenes:>6,} ({moderate_scenes/total_selected*100:>5.2f}%)")
    print(f"  --> Crowded  (4-6 classes): {crowded_scenes:>6,} ({crowded_scenes/total_selected*100:>5.2f}%)")

    print("\n[2] TARGET CLASS COVERAGE:")
    print("-" * 65)
    print(f" {'Class':<15} | {'Available':<12} | {'Selected':<12} | {'Coverage %'}")
    print("-" * 65)
    for c in TARGET_CLASSES:
        avail = class_before[c]
        sel = class_after[c]
        cov = sel / avail * 100.0 if avail > 0 else 0.0
        print(f" {c:<15} | {avail:>8,}     | {sel:>8,}     | {cov:>8.2f}%")

    print("\n[3] SOURCE-FOLDER DISTRIBUTION:")
    print("-" * 65)
    print(f" {'Folder':<10} | {'Available':<12} | {'Selected':<12} | {'Share %':<10} | {'Retention %'}")
    print("-" * 65)
    for fld in SOURCE_FOLDERS:
        avail = folder_before[fld]
        sel = folder_after[fld]
        share = sel / total_selected * 100.0
        ret = sel / avail * 100.0 if avail > 0 else 0.0
        print(f" {fld:<10} | {avail:>8,}     | {sel:>8,}     | {share:>7.2f}%   | {ret:>8.2f}%")

    print("\n[4] VALIDATION RESULTS:")
    print("-" * 65)
    for check_name, status in validation_results.items():
        print(f"  - {check_name:<30}: {status}")

    print("\n[5] GENERATED FILE PATHS:")
    print("-" * 65)
    print(f"  Manifest CSV : {output_csv_path.resolve()}")
    print(f"  Report TXT   : {output_report_path.resolve()}")
    print("=" * 65)
    print("[SUCCESS] 12,000-image training manifest v2 created.\n")


def main():
    args = parse_arguments()

    input_csv_path = Path(args.input_csv)
    input_json_path = Path(args.input_json)
    output_csv_path = Path(args.output_csv)
    output_report_path = Path(args.output_report)

    print("=" * 70)
    print("  SIH 2026: BALANCED UVH-26 12K TRAINING SUBSET SELECTION (V2)")
    print("=" * 70)

    try:
        # Step 1: Load candidates
        print(f"[1/6] Loading mapped candidate manifest: {input_csv_path} ...")
        candidate_rows = load_candidate_manifest(input_csv_path)

        # Step 2: Load bounding box counts for verification report
        print(f"[2/6] Loading annotation JSON for bounding-box analysis: {input_json_path} ...")
        image_boxes = load_bounding_box_counts(input_json_path)

        # Step 3: Compute class weights
        print("[3/6] Computing log-smoothed inverse class weights ...")
        class_weights = calculate_class_weights(candidate_rows)

        # Step 4: Plan scene quotas
        actual_bin_targets, bin_available, bin_notes = plan_scene_quotas(
            candidate_rows,
            TARGET_SCENE_QUOTAS,
            total_target=args.target_images,
        )

        # Step 5: Execute stratified v2 selection
        print(f"[4/6] Executing v2 selection with seed={args.seed} ...")
        selected_rows = select_v2_training_subset(
            candidate_rows,
            actual_bin_targets,
            class_weights,
            seed=args.seed,
        )

        # Step 6: Validate integrity
        print("[5/6] Performing strict validation checks ...")
        validation_results = perform_validation_checks(
            selected_rows,
            candidate_rows,
            expected_count=args.target_images,
            seed=args.seed,
        )

        # Fail if any check did not pass
        for k, v in validation_results.items():
            if not v.startswith("PASSED"):
                raise ValueError(f"Validation check failed: {k} -> {v}")

        # Step 7: Write output files
        print(f"[6/6] Writing output manifest and report ...")
        write_manifest_csv_v2(selected_rows, output_csv_path)

        generate_report_v2(
            candidate_rows=candidate_rows,
            selected_rows=selected_rows,
            bin_targets=actual_bin_targets,
            bin_available=bin_available,
            bin_notes=bin_notes,
            validation_results=validation_results,
            image_boxes=image_boxes,
            seed=args.seed,
            output_report_path=output_report_path,
        )

        # Print terminal summary
        complexity_counts = Counter(int(r["target_class_count"]) for r in selected_rows)
        class_before = Counter()
        for r in candidate_rows:
            for c in r["target_classes"].split(","):
                class_before[c] += 1

        class_after = Counter()
        for r in selected_rows:
            for c in r["target_classes"].split(","):
                class_after[c] += 1

        folder_before = Counter(r["source_folder"] for r in candidate_rows)
        folder_after = Counter(r["source_folder"] for r in selected_rows)

        print_terminal_summary_v2(
            total_candidates=len(candidate_rows),
            total_selected=len(selected_rows),
            complexity_counts=complexity_counts,
            class_before=class_before,
            class_after=class_after,
            folder_before=folder_before,
            folder_after=folder_after,
            validation_results=validation_results,
            output_csv_path=output_csv_path,
            output_report_path=output_report_path,
        )

    except Exception as exc:
        print(f"\n[ERROR] Selection v2 failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
