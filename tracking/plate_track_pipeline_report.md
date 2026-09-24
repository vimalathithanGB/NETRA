# NETRA Checkpoint 4: Video Pipeline + ByteTrack + License Plate OCR + Temporal Aggregation

**Date**: 2026-09-19  
**Component**: Checkpoint 4 — End-to-End Local Video Perception Pipeline  
**Module**: `tracking/plate_track_pipeline.py`  
**Project**: SIH 2026 NETRA (Networked Engine for Traffic Recognition & Analytics)  
**Status**: **COMPLETE & VERIFIED**

---

## 1. Architecture Overview

The video perception pipeline integrates vehicle detection, ByteTrack tracking, license plate detection, geometric association, throttled OCR, and confidence-weighted temporal voting:

```text
Video Frame (2560x1440)
  ├── GPU (RTX 3050): Model 1 (YOLOv8n UVH-26) ──> ByteTrack ──> Active Vehicle Tracks [ID, Class, BBox]
  └── GPU (RTX 3050): Model 2 (YOLOv8n Plate Detector) ──> License Plate BBoxes
         │
         ▼
  Geometric Association (Plate Center Containment / IoU)
         │
         ▼
  OCR Sampling Check (Throttle: Frame Δ >= 5 per track)
         │
         ▼
  CPU: Model 3 (PaddleOCR det=False, rec=True, cls=True) ──> Raw Text + Conf + Format Check
         │
         ▼
  Temporal OCR Buffer per Vehicle Track ID
         │
         ▼
  Confidence-Weighted Voting & Stability Evaluation
         │
         ├── Stable Plate Text (valid format, >= 3 obs, avg conf >= 0.70)
         ├── Tentative Plate Text (conf >= 0.50, < 3 valid obs)
         └── Unknown (no confident plate observation)
         │
         ▼
  Unified Track Record (JSON) + Annotated Surveillance Video (MP4)
```

---

## 2. Files Created & Updated

1. [tracking/plate_track_pipeline.py](file:///d:/apps/coding/hackathon%20projects/SIH%202026/SIH2026-AI-ENGINE/tracking/plate_track_pipeline.py):
   - **`PlateTrackPipeline`**: Orchestrator executing vehicle tracking, plate detection, geometric association, OCR throttling, and video rendering.
   - **`VehicleTrackState`**: Maintains track lifetime, bounding boxes, OCR observation history, and temporal voting state.
   - **`OCRObservation`**: Dataclass storing per-frame OCR prediction, cleaned text, confidence, and format validity.
2. [runs/pipeline/traffic_plate_tracking.mp4](file:///d:/apps/coding/hackathon%20projects/SIH%202026/SIH2026-AI-ENGINE/runs/pipeline/traffic_plate_tracking.mp4):
   - Annotated output surveillance video (32.65 MB, 228 frames, $2560 \times 1440$ at 29.97 FPS).
3. [runs/pipeline/traffic_plate_tracking.json](file:///d:/apps/coding/hackathon%20projects/SIH%202026/SIH2026-AI-ENGINE/runs/pipeline/traffic_plate_tracking.json):
   - Complete structured telemetry payload with metadata, hyperparameters, and per-track records.
4. [runs/pipeline/traffic_plate_tracking_report.md](file:///d:/apps/coding/hackathon%20projects/SIH%202026/SIH2026-AI-ENGINE/runs/pipeline/traffic_plate_tracking_report.md):
   - Benchmark and telemetry report for the run directory.
5. [tracking/plate_track_pipeline_report.md](file:///d:/apps/coding/hackathon%20projects/SIH%202026/SIH2026-AI-ENGINE/tracking/plate_track_pipeline_report.md):
   - Formal technical report for the tracking package.

---

## 3. Existing Modules Reused (Zero Redesign)

- **Vehicle Detector (Model 1)**: `runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/best.pt` (6 vehicle classes: car, motorcycle, auto_rickshaw, bus, truck, van).
- **ByteTrack Configuration**: `tracking/tracker_config.yaml` (Kalman motion filter, 2-stage association, track buffer 30 frames).
- **License Plate Detector (Model 2)**: `runs/detect/runs/plate_detection/vehicle_plate_yolov8n_50ep/weights/best.pt` (Single class: `license_plate`).
- **License Plate OCR (Model 3)**: `ocr/license_plate_ocr.py` (`LicensePlateOCR` on CPU, MKLDNN disabled, recognition-only SVTR_LCNet).
- **Vehicle Re-ID Extractor**: `reid/vehicle_reid.py` (verified functional, isolated from this checkpoint).

---

## 4. Plate-to-Vehicle Association Method

- For each detected plate box $[px_1, py_1, px_2, py_2]$:
  1. Calculate center point: $cx_p = \frac{px_1 + px_2}{2}, cy_p = \frac{py_1 + py_2}{2}$.
  2. Query active vehicle tracks where $vx_1 \le cx_p \le vx_2$ and $vy_1 \le cy_p \le vy_2$.
  3. If exactly 1 vehicle contains the center: associate with that vehicle's Track ID.
  4. If multiple vehicles contain the center: associate with the vehicle with the highest Intersection-over-Union (IoU).
  5. If 0 vehicles contain the center: mark unassociated (`associated_track_id = None`).

---

## 5. OCR Sampling Strategy

- OCR on CPU is computationally heavier than YOLO detection on GPU.
- Running OCR on every frame for every vehicle causes CPU starvation.
- **Strategy**: Throttled sampling interval $\Delta f \ge 5$ frames per track ID.
- Each vehicle is evaluated upon initial track confirmation and then every 5 frames, reducing CPU load by **$80\%$** while yielding sufficient samples for temporal aggregation.

---

## 6. Temporal Voting Algorithm

- For each Track ID, all OCR observations with confidence $\ge 0.50$ and `valid_format == True` cast votes.
- Each vote is weighted by the model's confidence:
  $$\text{Score}(\text{candidate}) = \sum_{i \in \text{valid}} \text{conf}_i \cdot \mathbb{I}(\text{text}_i = \text{candidate})$$
- The winning candidate is the string with the highest accumulated score.
- The reported plate confidence is the mean confidence of the winning candidate across its observations.

---

## 7. Stability Rules

- **Stable**: $\text{valid\_observation\_count} \ge 3$ AND $\text{plate\_confidence} \ge 0.70$.
- **Tentative**: OCR observation exists with confidence $\ge 0.50$, but valid observations $< 3$.
- **Unknown**: No confident OCR observations recorded.

---

## 8. Test Video Details

- **File**: `data/videos/test_2.mp4`
- **Resolution**: $2560 \times 1440$ (QHD)
- **Framerate**: 29.97 FPS
- **Duration**: 7.61 seconds (228 frames)
- **Scene**: Multi-lane urban highway traffic surveillance with dense vehicle flow.

---

## 9. Performance Results

| Performance Metric | Result |
|---|---|
| **Total Frames Processed** | **228 / 228 (100%)** |
| **Pipeline Wall-Clock Time** | **21.8 seconds** |
| **Average Processing Speed** | **10.4 FPS** |
| **GPU Detection Latency (Vehicle + Plate)** | **~35 ms / frame** (GPU) |
| **Average OCR Latency** | **230.8 ms / call** (CPU) |
| **GPU VRAM Utilization** | **~1.2 GB / 4.0 GB** (RTX 3050) |
| **OneDNN / C++ Engine Crashes** | **0 crashes** |

---

## 10. Track Statistics

| Metric | Count | Percentage |
|---|---:|---:|
| **Total Unique Vehicle Tracks** | **52** | 100.0% |
| **Stable Plate Tracks** | **0** | 0.0% |
| **Tentative Plate Tracks** | **2** | 3.8% |
| **Unknown Plate Tracks** | **50** | 96.2% |

*Note: In `test_2.mp4`, vehicles move rapidly across the camera field of view; plate detections were primarily concentrated on the large rear surfaces of heavy vehicles (Truck ID 39 and Car ID 45).*

---

## 11. OCR Statistics

| Metric | Value |
|---|---|
| **Plate Bounding-Box Detections** | 75 |
| **Plate-to-Vehicle Associations** | 26 |
| **OCR Invocations (Throttled)** | 9 |
| **Successful Character Extractions** | 9 (100%) |
| **Top Recognized String** | `"ONDUTY"` (Truck ID 39: conf 0.9681; Car ID 45: conf 0.9399) |
| **Format Validation Protection** | Correctly rejected `"ONDUTY"` from standard Indian plate format |

---

## 12. Known Limitations & Observations

1. **Camera Distance & Downscaling**:
   - In 2.5K wide-angle surveillance video, distant vehicle plates are under $30 \times 12$ pixels.
   - Plate detection requires high resolution inference (`imgsz=1280`) to resolve small plate boxes.
2. **Short Track Dwell Times**:
   - At 10.4 FPS across a 7.6-second clip, fast-moving vehicles remain in the camera view for 10–30 frames. With a 5-frame sampling interval, vehicles receive 2–6 OCR attempts.
3. **Signboards & Emergency Badges**:
   - Emergency or transport signs (such as "ON DUTY") mounted on vehicle tailgates are detected as plate-like rectangles. The regex format validator successfully prevented these from being marked as stable registration plates.

---

## 13. Exact Next Recommended Checkpoint

**Checkpoint 5: Unified Multi-Modal Vehicle Registry (ByteTrack + OSNet Re-ID + License Plate)**:
1. Integrate `reid/vehicle_reid.py` (OSNet-AIN 512-D embedding) with `tracking/plate_track_pipeline.py`.
2. Construct the full `VehicleTrackRecord`:
   - `track_id` (int)
   - `vehicle_class` (str)
   - `first_seen_frame`, `last_seen_frame` (int)
   - `reid_embedding` (512-float vector, L2-normalized)
   - `plate_text` (str)
   - `plate_status` (`stable` | `tentative` | `unknown`)
   - `plate_confidence` (float)
3. Prepare the unified data structure for downstream cross-camera re-identification and database persistence.
