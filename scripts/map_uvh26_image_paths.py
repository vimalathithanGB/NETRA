"""
SIH 2026 AI Engine - Dataset Preparation
=========================================
Script: scripts/map_uvh26_image_paths.py

Purpose:
    Maps every filename in 'datasets/UVH-26/uvh26_candidate_manifest.csv' to its
    actual remote path within the official UVH-26 Hugging Face dataset repository
    (iisc-aim/UVH-26).

Context for AI/ML Beginners:
    - Large computer vision datasets (like UVH-26, which is ~72 GB) are hosted on
      cloud model/dataset hubs (like Hugging Face Hub) using Git LFS (Large File Storage).
    - In the UVH-26 training set, thousands of high-resolution images are partitioned
      into subdirectories: 'UVH-26-Train/data/<folder>/<filename>.png' (e.g. folders 000 to 004)
      to avoid placing 20,000+ files into a single flat directory.
    - However, the COCO annotation JSON only records flat filenames (e.g. '356278.png').
    - Rather than downloading 72 GB of images just to find where each image lives, we use
      the Hugging Face Hub metadata API to inspect the repository tree remotely in seconds.
    - We then map each candidate image's filename to its exact folder ('source_folder')
      and full relative repository path ('source_path') and write the verified result into
      'datasets/UVH-26/uvh26_candidate_manifest_mapped.csv'.

Key Constraints:
    - Zero image downloads.
    - Zero dataset downloading.
    - Original JSON and existing manifest remain untouched.
    - Strict validation: checks for duplicate filenames, missing files, and 1-to-1 matches.
    - Pagination is handled gracefully via HfApi.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# We use the already-installed huggingface_hub package
from huggingface_hub import HfApi

# Target repository identifier on Hugging Face Hub
HF_REPO_ID = "iisc-aim/UVH-26"
HF_REPO_TYPE = "dataset"
TRAIN_DATA_PREFIX = "UVH-26-Train/data/"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with sensible repository defaults."""
    parser = argparse.ArgumentParser(
        description="Map UVH-26 candidate manifest filenames to Hugging Face repository paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="datasets/UVH-26/uvh26_candidate_manifest.csv",
        help="Path to the candidate manifest CSV generated in Phase 1.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="datasets/UVH-26/uvh26_candidate_manifest_mapped.csv",
        help="Path to save the mapped candidate manifest CSV.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help="Hugging Face dataset repository ID.",
    )
    return parser.parse_args()


def load_candidate_manifest(input_csv_path: Path) -> List[dict]:
    """
    Read the candidate manifest CSV.
    Returns a list of row dictionaries.
    """
    if not input_csv_path.exists():
        raise FileNotFoundError(
            f"Candidate manifest not found at: '{input_csv_path}'\n"
            f"Please ensure scripts/select_uvh26_subset.py has been run first."
        )

    rows: List[dict] = []
    with open(input_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[1/5] Loaded {len(rows):,} candidate records from: {input_csv_path}")
    return rows


def fetch_repository_file_list(api: HfApi, repo_id: str) -> List[str]:
    """
    Retrieve all file paths in the Hugging Face dataset repository matching:
    'UVH-26-Train/data/**/*.png'

    Pagination Handling:
        HfApi.list_repo_tree() is a generator that natively iterates through all
        paginated pages of the Git LFS tree. We also provide a fallback to
        HfApi.list_repo_files() for comprehensive reliability.
    """
    print(f"[2/5] Querying Hugging Face repository metadata for '{repo_id}' ...")
    print("      (Retrieving file tree index without downloading images)")

    all_matched_paths: List[str] = []

    try:
        # list_repo_tree natively handles pagination and yields RepoFile/RepoFolder objects
        tree_generator = api.list_repo_tree(
            repo_id=repo_id,
            repo_type=HF_REPO_TYPE,
            path_in_repo="UVH-26-Train/data",
            recursive=True,
        )

        for item in tree_generator:
            # item.path represents the relative path in the repo
            path_str = getattr(item, "path", None) or getattr(item, "rfilename", "")
            if path_str.startswith(TRAIN_DATA_PREFIX) and path_str.lower().endswith(".png"):
                all_matched_paths.append(path_str)

    except Exception as tree_err:
        print(f"      [INFO] list_repo_tree encountered: {tree_err}")
        print("      [INFO] Falling back to api.list_repo_files() ...")
        all_files = api.list_repo_files(repo_id=repo_id, repo_type=HF_REPO_TYPE)
        all_matched_paths = [
            f for f in all_files
            if f.startswith(TRAIN_DATA_PREFIX) and f.lower().endswith(".png")
        ]

    print(f"      Found {len(all_matched_paths):,} matching training PNG files in repository.")
    return all_matched_paths


def build_path_index(repo_paths: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Construct a mapping from filename -> repository path.
    Also detects any duplicate filenames across folders in the repository.

    Returns:
        unique_map: Dict[filename, repository_path] for unique entries
        duplicate_map: Dict[filename, List[paths]] for any filenames appearing in >1 folder
    """
    print("[3/5] Building filename -> path index and auditing for duplicates ...")

    filename_to_all_paths: Dict[str, List[str]] = defaultdict(list)

    for path in repo_paths:
        # Extract filename (e.g., '356278.png' from 'UVH-26-Train/data/000/356278.png')
        filename = path.split("/")[-1]
        filename_to_all_paths[filename].append(path)

    unique_map: Dict[str, str] = {}
    duplicate_map: Dict[str, List[str]] = {}

    for fname, paths in filename_to_all_paths.items():
        if len(paths) == 1:
            unique_map[fname] = paths[0]
        else:
            duplicate_map[fname] = paths

    return unique_map, duplicate_map


def map_and_save_manifest(
    manifest_rows: List[dict],
    unique_path_map: Dict[str, str],
    output_csv_path: Path,
) -> Tuple[int, List[str], Counter]:
    """
    Map each candidate manifest row to its remote source_path and source_folder.
    Writes the mapped data to output_csv_path.

    Returns:
        successfully_mapped_count: int
        missing_files: List[str] of filenames not found in repository index
        folder_counts: Counter of folder_name -> count of mapped candidate images
    """
    print(f"[4/5] Mapping candidate filenames and writing to: {output_csv_path} ...")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Columns: keep all original columns and include source_path & source_folder
    fieldnames = [
        "image_id",
        "file_name",
        "width",
        "height",
        "source_folder",
        "source_path",
        "target_classes",
        "target_class_count",
    ]

    successfully_mapped = 0
    missing_files: List[str] = []
    folder_counts: Counter = Counter()

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in manifest_rows:
            fname = row.get("file_name", "")
            matched_path = unique_path_map.get(fname)

            if matched_path:
                # Path format: 'UVH-26-Train/data/<folder>/<filename>'
                path_parts = matched_path.split("/")
                folder_name = path_parts[2] if len(path_parts) >= 3 else "UNKNOWN"

                # Update row dictionary
                row["source_folder"] = folder_name
                row["source_path"] = matched_path

                writer.writerow({
                    "image_id": row.get("image_id"),
                    "file_name": fname,
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "source_folder": folder_name,
                    "source_path": matched_path,
                    "target_classes": row.get("target_classes"),
                    "target_class_count": row.get("target_class_count"),
                })

                successfully_mapped += 1
                folder_counts[folder_name] += 1
            else:
                missing_files.append(fname)
                # Still record row with UNKNOWN paths if missing
                row["source_folder"] = "MISSING"
                row["source_path"] = "NOT_FOUND_IN_REPO"
                writer.writerow(row)

    return successfully_mapped, missing_files, folder_counts


def print_summary_report(
    total_candidates: int,
    mapped_count: int,
    missing_files: List[str],
    duplicate_map: Dict[str, List[str]],
    folder_counts: Counter,
    output_csv_path: Path,
):
    """
    Display a comprehensive, beginner-friendly verification report in terminal.
    """
    print("\n" + "=" * 75)
    print("       UVH-26 REMOTE PATH MAPPING - AUDIT & VERIFICATION REPORT")
    print("=" * 75)
    print(f" Target Repository              : {HF_REPO_ID} (repo_type='{HF_REPO_TYPE}')")
    print(f" Total Candidate Images         : {total_candidates:,}")
    print(f" Successfully Mapped to Paths   : {mapped_count:,} ({(mapped_count / total_candidates * 100):.2f}%)")
    print(f" Missing Files in Repository    : {len(missing_files)}")
    print(f" Duplicate Filenames in Repo    : {len(duplicate_map)}")

    # Display duplicate warnings if any exist
    if duplicate_map:
        print("\n [WARNING] Duplicate Filenames Detected in Repository:")
        for fname, paths in list(duplicate_map.items())[:10]:
            print(f"  - '{fname}' appears in multiple locations: {paths}")
        if len(duplicate_map) > 10:
            print(f"  ... and {len(duplicate_map) - 10} more duplicates.")
    else:
        print(" [AUDIT OK] Every repository filename is unique across all folders.")

    # Display missing files if any exist
    if missing_files:
        print("\n [WARNING] Candidate Files Missing from Repository:")
        for fname in missing_files[:10]:
            print(f"  - {fname}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more missing files.")

    # Detailed Folder Distribution
    print("-" * 75)
    print(" NUMBER OF IMAGES PER SOURCE FOLDER (Candidate Subset):")
    print("-" * 75)
    print(f" {'Folder Name':<15} | {'Candidate Images':<18} | {'Folder Share':<14}")
    print("-" * 75)

    # Folders of specific interest: 000, 001, 002, 003, 004
    target_folders = ["000", "001", "002", "003", "004"]
    for folder in target_folders:
        cnt = folder_counts.get(folder, 0)
        pct = (cnt / total_candidates * 100.0) if total_candidates > 0 else 0.0
        print(f"  {folder:<13} | {cnt:>12,} images   | {pct:>10.2f}%")

    # Check for any unexpected folders
    other_folders = [f for f in folder_counts if f not in target_folders]
    for folder in sorted(other_folders):
        cnt = folder_counts[folder]
        pct = (cnt / total_candidates * 100.0) if total_candidates > 0 else 0.0
        print(f"  {folder:<13} | {cnt:>12,} images   | {pct:>10.2f}%")

    file_size_kb = output_csv_path.stat().st_size / 1024.0 if output_csv_path.exists() else 0.0
    print("-" * 75)
    print(f" Output Mapped CSV              : {output_csv_path.resolve()}")
    print(f" Mapped CSV File Size           : {file_size_kb:.1f} KB")
    print("=" * 75)


def main():
    args = parse_arguments()

    input_csv_path = Path(args.input_csv)
    output_csv_path = Path(args.output_csv)

    print("=" * 75)
    print("  SIH 2026: UVH-26 HUGGING FACE PATH MAPPER")
    print("=" * 75)

    try:
        # Step 1: Read input manifest
        manifest_rows = load_candidate_manifest(input_csv_path)

        # Step 2: Query Hugging Face Hub metadata API (No image downloads!)
        api = HfApi()
        repo_paths = fetch_repository_file_list(api, repo_id=args.repo_id)

        # Step 3: Build filename -> path lookup & check duplicates
        unique_path_map, duplicate_map = build_path_index(repo_paths)

        if duplicate_map:
            print(f"\n[ALERT] Found {len(duplicate_map)} duplicate filenames in the repository!")
            print("Per requirements, duplicates will be reported and not silently chosen.\n")

        # Step 4: Map manifest rows to source_folder and source_path, write new CSV
        mapped_count, missing_files, folder_counts = map_and_save_manifest(
            manifest_rows,
            unique_path_map,
            output_csv_path,
        )

        # Step 5: Print detailed report and folder statistics
        print_summary_report(
            total_candidates=len(manifest_rows),
            mapped_count=mapped_count,
            missing_files=missing_files,
            duplicate_map=duplicate_map,
            folder_counts=folder_counts,
            output_csv_path=output_csv_path,
        )

        # Mandatory success confirmation specified in user request
        print("[SUCCESS] UVH-26 image-path mapping created.")

    except Exception as exc:
        print(f"\n[ERROR] Path mapping failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
