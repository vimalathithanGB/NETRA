"""
SIH 2026 AI Engine - Dataset Preparation
=========================================
Script: scripts/download_uvh26_training_set.py

Purpose:
    Selectively downloads ONLY the 12,000 training images designated in:
    datasets/UVH-26/uvh26_training_12000_manifest_v2.csv

    from the Hugging Face Hub dataset repository:
    iisc-aim/UVH-26

Key Features & Safeguards:
    1. Zero Unnecessary Downloads:
       Only downloads images listed in the 12,000-image manifest. Does NOT download
       the entire ~72 GB repository.
    2. Resume-Friendly (Skip Existing):
       If an image has already been downloaded and is non-empty, it is skipped instantly.
       You can safely interrupt and restart the download anytime.
    3. Robust Retry Mechanism:
       Retries failed downloads up to 3 times with exponential backoff before logging a failure.
    4. Fault Tolerant:
       A single failed download will not crash the script; it logs the failure and continues.
    5. Folder Preservation:
       Maintains the original UVH-26 partitioned folder structure:
       datasets/UVH-26/UVH-26-Train/data/000/
       datasets/UVH-26/UVH-26-Train/data/001/
       datasets/UVH-26/UVH-26-Train/data/002/
       datasets/UVH-26/UVH-26-Train/data/003/
       datasets/UVH-26/UVH-26-Train/data/004/
    6. Detailed Auditing:
       Generates both a summary report (uvh26_download_report.txt) and a failures CSV
       (uvh26_download_failures.csv).

Usage:
    python scripts/download_uvh26_training_set.py

    Optional Flags:
        --workers 8       (Use 8 parallel download threads for faster download; default=4)
        --limit 10        (Test download on just the first 10 images)
        --max-retries 5   (Adjust maximum retry attempts per failed image)
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

# We use the already-installed huggingface_hub package
from huggingface_hub import hf_hub_download

# Default repository configuration
DEFAULT_REPO_ID = "iisc-aim/UVH-26"
DEFAULT_REPO_TYPE = "dataset"
EXPECTED_MANIFEST_COUNT = 12000


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download only the 12,000 selected UVH-26 training images from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="datasets/UVH-26/uvh26_training_12000_manifest_v2.csv",
        help="Path to the 12,000-image training manifest v2 CSV.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--dest-root",
        type=str,
        default="datasets/UVH-26",
        help="Root local directory to unpack files (source_path will be appended to this).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="datasets/UVH-26/uvh26_download_report.txt",
        help="Path to save the final download report.",
    )
    parser.add_argument(
        "--failures",
        type=str,
        default="datasets/UVH-26/uvh26_download_failures.csv",
        help="Path to save failed download records if any occur.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download threads (1 = sequential, 4-8 = recommended for broadband).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per image before marking as failed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images to download (0 = download all records in manifest). Useful for test runs.",
    )
    return parser.parse_args()


def load_and_validate_manifest(manifest_path: Path, expected_count: int = EXPECTED_MANIFEST_COUNT) -> List[dict]:
    """
    Read the manifest and verify:
    1. File exists
    2. Exactly 12,000 records are present (if expected_count > 0)
    3. image_id uniqueness
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found at: '{manifest_path}'")

    rows: List[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_records = len(rows)
    print(f"[1/4] Loaded manifest: {manifest_path} ({total_records:,} records)")

    # Verify count
    if expected_count > 0 and total_records != expected_count:
        raise ValueError(
            f"Manifest validation failed: expected {expected_count:,} records, but found {total_records:,}."
        )

    # Verify image_id uniqueness
    image_ids = [r["image_id"] for r in rows]
    unique_ids = set(image_ids)
    if len(unique_ids) != total_records:
        duplicates = total_records - len(unique_ids)
        raise ValueError(f"Manifest validation failed: found {duplicates} duplicate image_ids!")

    print(f"      [OK] Verified {total_records:,} records with 100% unique image IDs.")
    return rows


def download_single_image(
    record: dict,
    dest_root: Path,
    repo_id: str,
    max_retries: int = 3,
) -> Tuple[str, str, Optional[str]]:
    """
    Download a single image file specified by record['source_path'].

    Returns:
        status: 'SKIPPED' | 'DOWNLOADED' | 'FAILED'
        source_path: path string
        error_message: error description if failed, else None
    """
    source_path = record["source_path"]
    local_path = dest_root / source_path

    # Check if image already exists and is non-empty
    if local_path.exists() and local_path.stat().st_size > 0:
        return "SKIPPED", source_path, None

    # Ensure local directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Retry loop
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=source_path,
                repo_type=DEFAULT_REPO_TYPE,
                local_dir=str(dest_root),
            )
            # Verify file was written and is non-empty
            if local_path.exists() and local_path.stat().st_size > 0:
                return "DOWNLOADED", source_path, None
            else:
                raise IOError(f"Downloaded file is missing or 0 bytes: {local_path}")
        except Exception as err:
            last_error = err
            if attempt < max_retries:
                time.sleep(1.0 * attempt)  # Exponential backoff

    error_msg = f"Failed after {max_retries} attempts: {last_error}"
    return "FAILED", source_path, error_msg


class ProgressTracker:
    """Thread-safe download progress tracker with console rendering."""

    def __init__(self, total: int):
        self.total = total
        self.downloaded = 0
        self.skipped = 0
        self.failed = 0
        self.processed = 0
        self.lock = Lock()
        self.start_time = time.time()
        self.last_print_time = 0.0

    def update(self, status: str):
        with self.lock:
            self.processed += 1
            if status == "DOWNLOADED":
                self.downloaded += 1
            elif status == "SKIPPED":
                self.skipped += 1
            elif status == "FAILED":
                self.failed += 1

            now = time.time()
            # Print update every 25 images, or on every failure, or at completion
            if (self.processed % 25 == 0) or (status == "FAILED") or (self.processed == self.total) or (now - self.last_print_time > 2.0):
                self.last_print_time = now
                self._render()

    def _render(self):
        elapsed = time.time() - self.start_time
        pct = (self.processed / self.total * 100.0) if self.total > 0 else 0.0
        speed = self.processed / elapsed if elapsed > 0 else 0.0

        sys.stdout.write(
            f"\r  Progress: [{self.processed:>6,}/{self.total:>6,}] ({pct:>5.1f}%) | "
            f"New: {self.downloaded:>5,} | Skipped: {self.skipped:>5,} | Failed: {self.failed:>3,} | "
            f"Speed: {speed:>5.1f} img/s"
        )
        sys.stdout.flush()

        if self.processed == self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def run_download_pipeline(
    records: List[dict],
    dest_root: Path,
    repo_id: str,
    workers: int = 4,
    max_retries: int = 3,
) -> Tuple[int, int, int, List[dict]]:
    """
    Execute download pool across records with worker threads.

    Returns:
        downloaded_count, skipped_count, failed_count, failed_records
    """
    total = len(records)
    tracker = ProgressTracker(total=total)
    failed_records: List[dict] = []

    print(f"[3/4] Starting download process ({workers} worker threads, max {max_retries} retries) ...")

    if workers <= 1:
        # Sequential execution
        for rec in records:
            status, path, err = download_single_image(rec, dest_root, repo_id, max_retries)
            if status == "FAILED":
                failed_entry = dict(rec)
                failed_entry["error_message"] = err
                failed_entry["attempts"] = max_retries
                failed_records.append(failed_entry)
            tracker.update(status)
    else:
        # Concurrent thread pool execution
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_rec = {
                executor.submit(download_single_image, rec, dest_root, repo_id, max_retries): rec
                for rec in records
            }

            for future in as_completed(future_to_rec):
                rec = future_to_rec[future]
                try:
                    status, path, err = future.result()
                except Exception as exc:
                    status = "FAILED"
                    err = str(exc)

                if status == "FAILED":
                    failed_entry = dict(rec)
                    failed_entry["error_message"] = err
                    failed_entry["attempts"] = max_retries
                    failed_records.append(failed_entry)

                tracker.update(status)

    return tracker.downloaded, tracker.skipped, tracker.failed, failed_records


def audit_final_available_files(records: List[dict], dest_root: Path) -> Tuple[int, Dict[str, int]]:
    """
    Count how many of the requested images now exist locally and are non-empty,
    categorized by source_folder.
    """
    available_count = 0
    folder_counts: Dict[str, int] = defaultdict(int)

    for r in records:
        local_path = dest_root / r["source_path"]
        if local_path.exists() and local_path.stat().st_size > 0:
            available_count += 1
            folder_counts[r["source_folder"]] += 1

    return available_count, dict(folder_counts)


def save_failures_csv(failed_records: List[dict], failures_path: Path):
    """Write failed downloads to CSV."""
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "file_name",
        "source_folder",
        "source_path",
        "error_message",
        "attempts",
    ]

    with open(failures_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in failed_records:
            writer.writerow({
                "image_id": rec.get("image_id"),
                "file_name": rec.get("file_name"),
                "source_folder": rec.get("source_folder"),
                "source_path": rec.get("source_path"),
                "error_message": rec.get("error_message", ""),
                "attempts": rec.get("attempts", 0),
            })


def save_download_report(
    total_requested: int,
    already_existing: int,
    newly_downloaded: int,
    failed_count: int,
    final_available: int,
    folder_available: Dict[str, int],
    report_path: Path,
    failures_path: Path,
    repo_id: str,
    duration_sec: float,
):
    """Generate human-readable download audit report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 75,
        "SIH 2026 AI ENGINE - UVH-26 12K TRAINING SET DOWNLOAD REPORT",
        "=" * 75,
        f"Timestamp                    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source Repository            : {repo_id}",
        f"Execution Time               : {duration_sec:.1f} seconds",
        "",
        "-" * 75,
        "DOWNLOAD SUMMARY METRICS",
        "-" * 75,
        f"Total Images Requested       : {total_requested:,}",
        f"Already Existing (Skipped)   : {already_existing:,}",
        f"Newly Downloaded             : {newly_downloaded:,}",
        f"Failed Downloads             : {failed_count:,}",
        f"Final Local Available Files  : {final_available:,} / {total_requested:,} ({(final_available / total_requested * 100):.2f}%)",
        "",
        "-" * 75,
        "AVAILABLE IMAGES BY SOURCE FOLDER",
        "-" * 75,
        f" {'Folder':<12} | {'Available Files':<18} | {'Folder Share':<14}",
        "-" * 75,
    ]

    for fld in ["000", "001", "002", "003", "004"]:
        cnt = folder_available.get(fld, 0)
        pct = (cnt / final_available * 100.0) if final_available > 0 else 0.0
        lines.append(f"  {fld:<10} | {cnt:>12,} files   | {pct:>10.2f}%")

    lines.extend([
        "-" * 75,
        "",
        "-" * 75,
        "FAILURES & INTEGRITY",
        "-" * 75,
        f"Total Failures               : {failed_count}",
        f"Failures Log CSV             : {failures_path.resolve() if failed_count > 0 else 'None (0 failures)'}",
        "",
        "=" * 75,
        "END OF DOWNLOAD REPORT",
        "=" * 75,
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_arguments()

    manifest_path = Path(args.manifest)
    dest_root = Path(args.dest_root)
    report_path = Path(args.report)
    failures_path = Path(args.failures)

    print("=" * 75)
    print("  SIH 2026: UVH-26 12K TRAINING SET SELECTIVE DOWNLOADER")
    print("=" * 75)

    start_time = time.time()

    # Step 1: Read and validate manifest
    records = load_and_validate_manifest(
        manifest_path,
        expected_count=EXPECTED_MANIFEST_COUNT if args.limit <= 0 else 0,
    )

    if args.limit > 0:
        print(f"      [LIMIT] User specified --limit {args.limit}; downloading only first {args.limit} images.")
        records = records[:args.limit]

    # Step 2: Ensure destination directory exists
    train_data_dir = dest_root / "UVH-26-Train" / "data"
    train_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/4] Target directory verified: {train_data_dir.resolve()}")

    # Step 3: Run download pipeline
    newly_downloaded, already_existing, failed_count, failed_records = run_download_pipeline(
        records=records,
        dest_root=dest_root,
        repo_id=args.repo_id,
        workers=args.workers,
        max_retries=args.max_retries,
    )

    # Step 4: Audit final available files
    print("[4/4] Auditing final local available files ...")
    final_available, folder_available = audit_final_available_files(records, dest_root)

    duration = time.time() - start_time

    # Save failures CSV
    save_failures_csv(failed_records, failures_path)

    # Save download report
    save_download_report(
        total_requested=len(records),
        already_existing=already_existing,
        newly_downloaded=newly_downloaded,
        failed_count=failed_count,
        final_available=final_available,
        folder_available=folder_available,
        report_path=report_path,
        failures_path=failures_path,
        repo_id=args.repo_id,
        duration_sec=duration,
    )

    # Final terminal summary
    print("\n" + "=" * 75)
    print("                 UVH-26 DOWNLOAD SUMMARY REPORT")
    print("=" * 75)
    print(f" Total Requested Images       : {len(records):,}")
    print(f" Already Existing (Skipped)   : {already_existing:,}")
    print(f" Newly Downloaded             : {newly_downloaded:,}")
    print(f" Failed Downloads             : {failed_count:,}")
    print(f" Final Local Available Files  : {final_available:,} ({(final_available / len(records) * 100):.2f}%)")
    print("-" * 75)
    print(" Files Available by Folder:")
    for fld in ["000", "001", "002", "003", "004"]:
        cnt = folder_available.get(fld, 0)
        print(f"  - Folder {fld}: {cnt:>6,} images")
    print("-" * 75)
    print(f" Download Report Saved To     : {report_path.resolve()}")
    if failed_count > 0:
        print(f" Failures Log Saved To        : {failures_path.resolve()}")
    print(f" Execution Duration           : {duration:.1f} seconds")
    print("=" * 75)

    if failed_count == 0:
        print("\n[SUCCESS] UVH-26 12K training subset downloaded successfully with 0 failures.\n")
    else:
        print(f"\n[WARNING] Completed with {failed_count} failures. Inspect {failures_path} for details.\n")


if __name__ == "__main__":
    main()
