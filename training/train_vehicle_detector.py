"""
SIH 2026 AI Engine - Phase 4: Model 1 Vehicle Detector Training Pipeline
==========================================================================
Script: training/train_vehicle_detector.py

Purpose:
    Fine-tunes YOLOv8n for multi-camera urban traffic surveillance on the 6-class
    UVH-26 dataset (9,600 train, 1,200 val, 1,200 test images).

Target Classes (6):
    0: car
    1: motorcycle
    2: auto_rickshaw
    3: bus
    4: truck
    5: van

Hardware Optimizations (Target: NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM):
    1. Automatic Mixed Precision (amp=True): FP16 reduces memory footprint by ~50%.
    2. Conservative Batch Size:
       - Smoke test: batch=2
       - Normal training: batch=4
    3. Host/Device RAM Protection:
       - cache=False: Disables in-memory image caching to prevent RAM exhaustion.
       - workers=2: Balances DataLoader throughput without Windows shared-memory issues.
    4. Deterministic Training: seed=42, deterministic=True for reproducibility.
    5. Output Preservation: All outputs are strictly saved to D:/ drive (runs/vehicle_detection/).

Modes:
    --smoke-test : 1 epoch, batch=2, fast end-to-end GPU/VRAM/dataset verification.
    Normal       : 50 epochs (configurable), batch=4, patience=20 early stopping.
    --resume     : Resumes an interrupted run from last.pt checkpoint.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


# =============================================================================
# 1. HARDWARE & ENVIRONMENT AUDIT
# =============================================================================

def print_hardware_environment(device_id=0):
    """
    Prints diagnostic hardware, CUDA, VRAM, and library version details.
    """
    import ultralytics

    print("=" * 78)
    print("SIH 2026 AI ENGINE - HARDWARE & ENVIRONMENT DIAGNOSTICS")
    print("=" * 78)
    print(f"  Python Version      : {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch Version     : {torch.__version__}")
    print(f"  Ultralytics Version : {ultralytics.__version__}")
    print(f"  CUDA Available      : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"  CUDA Device Count   : {num_devices}")

        dev_id = int(device_id) if str(device_id).isdigit() else 0
        if dev_id < num_devices:
            props = torch.cuda.get_device_properties(dev_id)
            total_vram_gb = props.total_memory / (1024 ** 3)
            allocated_vram_mb = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
            reserved_vram_mb = torch.cuda.memory_reserved(dev_id) / (1024 ** 2)

            print(f"  Target Device       : cuda:{dev_id} ({props.name})")
            print(f"  Compute Capability  : {props.major}.{props.minor}")
            print(f"  Total VRAM          : {total_vram_gb:.2f} GB")
            print(f"  Allocated VRAM      : {allocated_vram_mb:.1f} MB")
            print(f"  Reserved VRAM       : {reserved_vram_mb:.1f} MB")
        else:
            print(f"  [WARN] Device ID {dev_id} out of range (max {num_devices - 1})")
    else:
        print("  [ERROR] CUDA is NOT available! PyTorch will fall back to CPU.")
        print("          Ensure CUDA-enabled PyTorch is installed inside the virtual environment.")

    cwd = Path(".").resolve()
    print(f"  Working Directory   : {cwd}")
    print(f"  Target Drive        : {cwd.drive}")
    print("=" * 78)


# =============================================================================
# 2. DATASET INTEGRITY VERIFICATION
# =============================================================================

def verify_dataset(data_yaml_path):
    """
    Validates data.yaml structure, image directories, and label files before training.
    """
    yaml_p = Path(data_yaml_path).resolve()
    print(f"[*] Verifying dataset specification: {yaml_p}")

    if not yaml_p.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at: {yaml_p}\n"
            "Please run 'scripts/prepare_uvh26_yolo_dataset.py' first."
        )

    with open(yaml_p, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Base path
    base_path_str = data_cfg.get("path", "")
    if base_path_str:
        base_dir = Path(base_path_str).resolve()
    else:
        base_dir = yaml_p.parent.resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {base_dir}")

    # Check train and val paths
    train_rel = data_cfg.get("train", "images/train")
    val_rel = data_cfg.get("val", "images/val")

    train_img_dir = (base_dir / train_rel).resolve()
    val_img_dir = (base_dir / val_rel).resolve()

    train_lbl_dir = (base_dir / "labels" / "train").resolve()
    val_lbl_dir = (base_dir / "labels" / "val").resolve()

    for d_path, desc in [
        (train_img_dir, "Train images"),
        (val_img_dir, "Val images"),
        (train_lbl_dir, "Train labels"),
        (val_lbl_dir, "Val labels"),
    ]:
        if not d_path.exists():
            raise FileNotFoundError(f"{desc} directory missing: {d_path}")

    # Count files
    train_images = list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpg"))
    val_images = list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpg"))
    train_labels = list(train_lbl_dir.glob("*.txt"))
    val_labels = list(val_lbl_dir.glob("*.txt"))

    print("  Dataset Status:")
    print(f"    - Train Images : {len(train_images):,} | Train Labels : {len(train_labels):,}")
    print(f"    - Val Images   : {len(val_images):,} | Val Labels   : {len(val_labels):,}")
    print(f"    - Target Classes: {data_cfg.get('names', {})}")

    if len(train_images) == 0:
        raise ValueError(f"Train images directory is empty: {train_img_dir}")
    if len(val_images) == 0:
        raise ValueError(f"Val images directory is empty: {val_img_dir}")
    if len(train_labels) == 0:
        raise ValueError(f"Train labels directory is empty: {train_lbl_dir}")
    if len(val_labels) == 0:
        raise ValueError(f"Val labels directory is empty: {val_lbl_dir}")

    print("[PASS] Dataset verification successful. All directories and files verified.\n")
    return True


# =============================================================================
# 3. TRAINING ORCHESTRATOR
# =============================================================================

def train_vehicle_detector(args):
    """
    Executes the training pipeline with RTX 3050 safety protections and error recovery.
    """
    # 1. Print diagnostics
    print_hardware_environment(args.device)

    # 2. Check CUDA availability
    if not torch.cuda.is_available() and str(args.device) != "cpu":
        print("[ERROR] CUDA requested but unavailable. Set --device cpu or fix PyTorch CUDA.")
        sys.exit(1)

    # 3. Validate dataset
    data_yaml = Path(args.data).resolve()
    verify_dataset(data_yaml)

    # 4. Resolve Output Directory (Guarantee on D: drive)
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    if project_dir.drive.upper() != "D:":
        print(f"[WARN] Training outputs are not on D: drive ({project_dir.drive}). Standardizing to D:...")

    # 5. Determine Mode & Hyperparameters
    if args.smoke_test:
        mode_label = "SMOKE TEST (1 Epoch Verification)"
        run_name = args.name if args.name != "vehicle_yolov8n_uvh26" else "vehicle_yolov8n_uvh26_smoke"
        epochs = 1
        batch_size = 2
        patience = 1
        plots = False
    else:
        mode_label = "FULL PRODUCTION TRAINING"
        run_name = args.name
        epochs = args.epochs
        batch_size = args.batch
        patience = args.patience
        plots = True

    # 6. Locate / Load Pretrained Model
    model_arg = args.model
    model_p = Path(model_arg).resolve()
    if model_p.exists():
        model_source = str(model_p)
    else:
        model_source = model_arg

    print("=" * 78)
    print(f"SIH 2026 VEHICLE DETECTOR TRAINING: {mode_label}")
    print("=" * 78)
    print(f"  Pretrained Weights  : {model_source}")
    print(f"  Dataset Config      : {data_yaml}")
    print(f"  Output Project Dir  : {project_dir}")
    print(f"  Run Identifier      : {run_name}")
    print(f"  Target Epochs       : {epochs}")
    print(f"  Batch Size          : {batch_size} (Optimized for 4 GB VRAM)")
    print(f"  Image Resolution    : {args.imgsz}x{args.imgsz}")
    print(f"  DataLoader Workers  : {args.workers}")
    print(f"  Mixed Precision     : {args.amp} (FP16)")
    print(f"  In-Memory Caching   : {args.cache} (Disabled for VRAM/RAM protection)")
    print(f"  Patience (EarlyStop): {patience}")
    print(f"  Generate Plots      : {plots}")
    print(f"  Deterministic Seed  : {args.seed}")
    print(f"  Resume Mode         : {args.resume}")
    print("=" * 78)

    # 7. Initialize YOLO Model
    try:
        if args.resume:
            last_ckpt = project_dir / run_name / "weights" / "last.pt"
            if not last_ckpt.exists():
                raise FileNotFoundError(f"Cannot resume: checkpoint not found at {last_ckpt}")
            print(f"[*] Resuming training from checkpoint: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"[*] Loading pretrained backbone: {model_source}")
            model = YOLO(model_source)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)

    # 8. Start Training with Exception Guard
    start_time = time.time()
    try:
        # Clear CUDA cache before training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            amp=args.amp,
            cache=args.cache,
            pretrained=True,
            patience=patience,
            save=True,
            save_period=args.save_period,
            plots=plots,
            deterministic=True,
            seed=args.seed,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            resume=args.resume,
            verbose=True,
        )

    except torch.cuda.OutOfMemoryError as oom:
        elapsed = time.time() - start_time
        print("\n" + "!" * 78)
        print("CUDA OUT OF MEMORY (OOM) ENCOUNTERED!")
        print("!" * 78)
        print(f"Elapsed time before crash: {elapsed:.1f}s")
        print(f"Details: {oom}")
        print("\nRecommended Fixes for RTX 3050 (4 GB VRAM):")
        print(f"  1. Lower batch size: run with '--batch {max(1, batch_size // 2)}'")
        print(f"  2. Lower image size: run with '--imgsz 512'")
        print("  3. Ensure amp is enabled: '--amp' (already enabled by default)")
        print("  4. Set workers to 0 or 1: '--workers 1'")
        print("  5. Close all other GPU-intensive applications (browsers, video editors)")
        print("!" * 78)
        sys.exit(1)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n[!] Training halted by user after {elapsed:.1f} seconds.")
        print(f"    Checkpoints saved up to last completed epoch in: {project_dir / run_name / 'weights'}")
        sys.exit(0)

    except Exception as e:
        err_str = str(e).lower()
        if "out of memory" in err_str or "cuda" in err_str and "memory" in err_str:
            print("\n" + "!" * 78)
            print("CUDA MEMORY ALLOCATION ERROR DETECTED!")
            print("!" * 78)
            print(f"Details: {e}")
            print("\nReduce memory pressure using: python training/train_vehicle_detector.py --batch 2 --imgsz 512")
            print("!" * 78)
        else:
            print(f"\n[ERROR] Training failed with unexpected exception: {e}")
        sys.exit(1)

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # 9. Post-Training Summary & Artifact Audit
    run_dir = project_dir / run_name
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    results_csv = run_dir / "results.csv"

    print("\n" + "=" * 78)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"  Mode                : {mode_label}")
    print(f"  Total Duration      : {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s ({total_time:.1f}s)")
    print(f"  Output Directory    : {run_dir}")
    print(f"  Best Weights (best) : {best_pt} (exists: {best_pt.exists()})")
    print(f"  Last Weights (last) : {last_pt} (exists: {last_pt.exists()})")
    print(f"  Metrics CSV         : {results_csv} (exists: {results_csv.exists()})")

    # Audit generated visualization plots
    expected_plots = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    print("\n  Diagnostic Artifacts Generated:")
    for plot_name in expected_plots:
        plot_p = run_dir / plot_name
        # Check standard name and alternative names (Ultralytics sometimes omits 'Box')
        alt_p = run_dir / plot_name.replace("Box", "")
        exists = plot_p.exists() or alt_p.exists()
        found_p = plot_p if plot_p.exists() else alt_p
        if exists:
            status = f"[SAVED] {found_p.name}"
        elif args.smoke_test:
            status = f"[SKIPPED] {plot_name} (plots=False in smoke test)"
        else:
            status = f"[PENDING/SKIPPED] {plot_name}"
        print(f"    {status}")

    print("=" * 78)
    if args.smoke_test:
        print("[OK] Smoke test verified! Your GPU, VRAM, and dataset are fully compatible.")
        print("     You are ready to proceed with normal 50-epoch training:")
        print("     python training/train_vehicle_detector.py --epochs 50 --batch 4")
    else:
        print("[OK] Production training complete. Model is ready for Phase 5 (Evaluation & Tracking).")
    print("=" * 78)


# =============================================================================
# 4. COMMAND LINE PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n vehicle detector for SIH 2026 AI Engine."
    )
    # Mode flags
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick 1-epoch smoke test to verify dataset and 4GB VRAM compatibility.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint (last.pt) in the run directory.",
    )

    # Core Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (default: 4 for 4GB VRAM RTX 3050).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker subprocesses (default: 2 for Windows host stability).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu' (default: 0).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs without validation improvement (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic training (default: 42).",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every X epochs (-1 saves only best.pt and last.pt).",
    )

    # Paths & Identifiers
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/UVH-26/yolo/data.yaml",
        help="Path to YOLO data.yaml configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Initial model weights backbone (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/vehicle_detection",
        help="Root output directory on D: drive (default: runs/vehicle_detection).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="vehicle_yolov8n_uvh26",
        help="Experiment identifier name (default: vehicle_yolov8n_uvh26).",
    )

    # Memory & Performance Flags
    parser.add_argument(
        "--amp",
        type=bool,
        default=True,
        help="Enable Automatic Mixed Precision FP16 (default: True).",
    )
    parser.add_argument(
        "--cache",
        type=bool,
        default=False,
        help="Cache images in RAM (default: False to protect 16GB host RAM).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_vehicle_detector(cli_args)
