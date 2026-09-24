# Phase 4: Vehicle Detector Training Guide (Model 1 — YOLOv8n)

This directory contains the training pipeline for **Model 1: Vehicle Detection** in the **SIH 2026 AI Engine** project. The model is fine-tuned on the 6-class UVH-26 dataset across 12,000 traffic surveillance scenes.

---

## 1. Hardware Profile & Target Constraints

| Parameter | Value | Rationale |
|---|---|---|
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU | 4 GB GDDR6 VRAM |
| **System RAM** | 16 GB DDR4/DDR5 | Host memory |
| **Operating System** | Windows 11 | Project directory on drive `D:` |
| **PyTorch & CUDA** | PyTorch 2.6.0 + CUDA 12.4 | GPU accelerated |
| **Ultralytics YOLO** | Ultralytics 8.4.152 | YOLOv8n backbone |
| **Dataset Storage** | 37.47 GB original PNGs | Zero duplication via NTFS hardlinks |

---

## 2. Recommended RTX 3050 (4 GB VRAM) Hyperparameters

Due to the **4 GB VRAM boundary**, the pipeline employs specific memory-safe settings:

* **Batch Size (`batch=4`)**: Keeps VRAM allocation safely around **~2.8–3.2 GB**, avoiding CUDA Out-of-Memory (OOM) crashes during backpropagation.
* **Automatic Mixed Precision (`amp=True`)**: Uses FP16 operations on RTX Tensor Cores, cutting gradient memory usage by ~50% while accelerating forward/backward passes.
* **In-Memory Image Caching (`cache=False`)**: Disabled to prevent filling the 16 GB host RAM or triggering paging to disk.
* **DataLoader Workers (`workers=2`)**: Balances image pre-fetching without spawning excessive multiprocessing handles on Windows.
* **Image Resolution (`imgsz=640`)**: Standard YOLO input resolution.

---

## 3. Step 1: Running the Smoke Test (Mandatory First Step)

Before launching a long multi-hour training run, run the **1-epoch smoke test** to verify:
1. PyTorch CUDA GPU allocation without OOM.
2. Dataset loader reads hardlinked images and `.txt` labels correctly.
3. Loss calculation, backward pass, and validation evaluation complete with 0 errors.

### Command:
```powershell
.\venv\Scripts\python.exe training/train_vehicle_detector.py --smoke-test
```

### Smoke Test Configuration:
* **Epochs**: `1`
* **Batch Size**: `2`
* **Resolution**: `640x640`
* **Workers**: `2`
* **Plots**: `False` (bypasses plotting library during quick sanity verification)
* **Output Run**: `runs/vehicle_detection/vehicle_yolov8n_uvh26_smoke/`

---

## 4. Step 2: Running Normal Production Training

Once the smoke test completes with `[OK]`, launch the full production training run:

### Default Command (50 Epochs, Batch 4):
```powershell
.\venv\Scripts\python.exe training/train_vehicle_detector.py --epochs 50 --batch 4
```

### Optional Command-Line Overrides:
```powershell
# Custom epochs or batch size
.\venv\Scripts\python.exe training/train_vehicle_detector.py --epochs 60 --batch 4 --imgsz 640

# Extra conservative mode (if system has heavy background tasks running)
.\venv\Scripts\python.exe training/train_vehicle_detector.py --epochs 50 --batch 2 --imgsz 640 --workers 1
```

---

## 5. How to Resume Interrupted Training

If training is stopped (e.g. laptop sleep, reboot, user interruption with `Ctrl+C`), you can seamlessly resume from the last saved epoch without starting from scratch:

```powershell
.\venv\Scripts\python.exe training/train_vehicle_detector.py --resume
```

Ultralytics will automatically reload `runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/last.pt`, restore the optimizer state, learning rate scheduler, and epoch counter, and continue training to completion.

---

## 6. Expected Output Artifacts

All training outputs are saved strictly on the **`D:` drive** under:
```text
D:\apps\coding\hackathon projects\SIH 2026\SIH2026-AI-ENGINE\runs\vehicle_detection\vehicle_yolov8n_uvh26\
├── weights/
│   ├── best.pt                     # Checkpoint with highest validation mAP50-95
│   └── last.pt                     # Checkpoint from the most recent epoch (for resume)
├── results.csv                     # Per-epoch training/val loss and mAP metrics
├── results.png                     # Loss curves, mAP50, and mAP50-95 plots
├── confusion_matrix.png            # Multi-class confusion matrix
├── confusion_matrix_normalized.png # Normalized confusion matrix
├── BoxPR_curve.png                 # Precision-Recall curve across all 6 classes
├── BoxF1_curve.png                 # F1-score vs Confidence curve
├── BoxP_curve.png & BoxR_curve.png # Precision and Recall curves
├── val_batch0_labels.jpg           # Ground truth annotations on validation batch
├── val_batch0_pred.jpg             # Model predictions on validation batch
└── args.yaml                       # Full record of hyperparameters and training configuration
```

---

## 7. Troubleshooting & Recovery

### CUDA Out of Memory (OOM)
If you encounter `torch.cuda.OutOfMemoryError`:
1. Reduce batch size to 2: `--batch 2`.
2. Reduce resolution to 512: `--imgsz 512`.
3. Set DataLoader workers to 1: `--workers 1`.
4. Close GPU-accelerated applications (Chrome, Edge, Discord, video players).

### Missing Dataset Error
If the script alerts that `data.yaml` or directories are missing:
```powershell
# Re-run Phase 3 preparation script
.\venv\Scripts\python.exe scripts/prepare_uvh26_yolo_dataset.py
```
