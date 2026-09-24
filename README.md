# SIH 2026: Multi-Camera Traffic Surveillance & Vehicle Re-ID AI Engine

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Deep Learning Framework](https://img.shields.io/badge/Framework-PyTorch_CUDA-orange.svg)](https://pytorch.org/)
[![Detection](https://img.shields.io/badge/Detection-Ultralytics_YOLO-00FFFF.svg)](https://docs.ultralytics.com/)
[![Project Status](https://img.shields.io/badge/Phase_2-Pretrained_YOLO_Inference-brightgreen.svg)](#development-phases)

---

## 1. Project Objective

The **SIH 2026 AI Engine** is a modular, software-only, edge-capable artificial intelligence engine designed for intelligent multi-camera traffic surveillance, automated vehicle tracking, license plate recognition, and cross-camera vehicle trajectory reconstruction.

Modern traffic monitoring faces severe challenges across non-overlapping, distributed camera networks:
* Vehicles disappear from one camera view and reappear in another with altered angles, lighting, and scale.
* License plates may be obscured, dirty, motion-blurred, or missing.
* Monolithic AI models fail to process multiple 30 FPS video streams in real time.

This project solves these challenges by combining specialized, high-throughput computer vision models into an incremental, deterministic perception pipeline, while leveraging visual-language reasoning (VLM) solely as an asynchronous cognitive layer.

### Key Capabilities
1. **Vehicle Detection & Classification**: Real-time detection of cars, trucks, buses, motorcycles, and auto-rickshaws.
2. **License Plate Detection**: High-precision localization of number plates under complex angles and lighting.
3. **Number Plate OCR**: Alphanumeric extraction (ANPR/ALPR) resilient to Indian vehicle plate variations.
4. **Single-Camera Multi-Object Tracking**: Persistent intra-camera trajectory and ID maintenance (ByteTrack/BoT-SORT).
5. **Vehicle Re-Identification (Re-ID)**: Deep visual appearance feature extraction (OSNet) invariant to viewpoint changes.
6. **Cross-Camera Vehicle Matching**: Spatio-temporal association of vehicles transitioning between camera zones.
7. **Trajectory Reconstruction**: Rebuilding continuous multi-camera physical routes on a global time-space map.
8. **Traffic Event & Rule Analysis**: Detection of wrong-way driving, speeding, illegal turns, and congestion.
9. **Optional VLM/LLM Reasoning**: Natural language scene auditing, incident summary generation, and edge-case dispute resolution (e.g., Qwen3-VL).
10. **FastAPI Telemetry & Web Dashboard Layer**: High-speed REST and WebSocket communication to interface with modern frontend dashboards.

---

## 2. Complete AI Pipeline Architecture

The perception engine processes multi-camera video streams sequentially through specialized stages:

```
+-----------------------------------------------------------------------------------+
|                            MULTI-CAMERA VIDEO FEEDS                               |
|                Camera 01 (North)     Camera 02 (South)     Camera 03 (East)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STAGE 1: INGESTION & VEHICLE DETECTION (Ultralytics YOLO)                         |
|   - Real-time frame extraction                                                    |
|   - Multi-class bounding box detection: [x1, y1, x2, y2, conf, class_id]          |
+-----------------------------------------------------------------------------------+
             |                                              |
             v                                              v
+------------------------------------+    +-----------------------------------------+
| STAGE 2: NUMBER PLATE DETECTION   |    | STAGE 3: SINGLE-CAMERA TRACKING         |
|   - Crop vehicle region of interest|    |   - ByteTrack / BoT-SORT Kalman Filter   |
|   - High-resolution plate bbox     |    |   - Single-camera Track ID assignment   |
|   - Perspective rectification      |    |   - Motion trajectory vector estimation |
+------------------------------------+    +-----------------------------------------+
             |                                              |
             v                                              v
+------------------------------------+    +-----------------------------------------+
| STAGE 4: NUMBER PLATE OCR          |    | STAGE 5: DEEP FEATURE RE-ID (OSNet)     |
|   - Character localization         |    |   - Crop vehicle appearance image       |
|   - Text recognition (PaddleOCR)   |    |   - 512-dim visual embedding generation |
|   - Regular expression syntax clean|    |   - Normalized feature vector           |
+------------------------------------+    +-----------------------------------------+
             \                                              /
              \----------------------\  /------------------/
                                     v  v
+-----------------------------------------------------------------------------------+
| STAGE 6: CROSS-CAMERA SPATIO-TEMPORAL MATCHING ENGINE                             |
|   - Multi-modal association:                                                      |
|       * Visual similarity (Cosine distance on OSNet embeddings)                   |
|       * Plate identity matching (Levenshtein distance on OCR strings)             |
|       * Spatio-temporal feasibility (transit time between Camera A and Camera B)  |
|   - Global Vehicle ID Assignment across non-overlapping views                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STAGE 7: TRAJECTORY RECONSTRUCTION & EVENT ANALYSIS                               |
|   - Continuous physical route timeline reconstruction                             |
|   - Traffic rule violations (Speed estimation, wrong-way, red-light, illegal turn)|
+-----------------------------------------------------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
+-----------------------------------------+  +--------------------------------------+
| [OPTIONAL] STAGE 8: VLM REASONING LAYER |  | STAGE 9: API & DASHBOARD STREAMING   |
|   - Triggered on flagged incidents      |  |   - FastAPI REST endpoints           |
|   - Visual Language Model (Qwen3-VL)    |  |   - WebSocket live alerts            |
|   - Natural language audit explanation  |  |   - Web Dashboard interface          |
+-----------------------------------------+  +--------------------------------------+
```

---

## 3. Role of Each Model & Component

| Component | Technology / Model | Primary Role |
| :--- | :--- | :--- |
| **Vehicle Detection** | YOLOv8 / YOLOv11 | Fast, frame-by-frame object localization (bounding boxes and class labels for vehicles). |
| **Plate Detection** | YOLOv8-Plate (Custom) | Dedicated micro-detector to locate license plates even on distant or angled vehicles. |
| **Number Plate OCR** | PaddleOCR (SVTR / CRNN) | High-accuracy character recognition capable of reading distorted and low-light plates. |
| **Intra-Camera Tracking**| ByteTrack / BoT-SORT | Associates detections across consecutive frames to maintain persistent single-camera track IDs. |
| **Vehicle Re-ID** | OSNet (`osnet_x0_25`) | Extracts a compact 512-dimensional visual embedding representing color, shape, and unique marks. |
| **Cross-Camera Matcher** | Spatio-Temporal Graph Engine | Fuses visual embeddings, OCR text, and camera topology (distance/time) to track vehicles across cameras. |
| **Trajectory Engine** | Mathematical Interpolator | Plots vehicle transit timestamps and spatial coordinates along the street network. |
| **VLM Cognitive Layer** | Qwen3-VL / Multimodal Transformers | Asynchronous agent that reviews flagged video clips to provide natural language explanations and resolve ambiguities. |
| **Backend & API** | FastAPI + WebSockets | Exposes live alerts, track histories, search endpoints, and telemetry to frontend clients. |

---

## 4. Why Specialized Computer Vision Models Instead of One End-to-End LLM?

A central design principle of this project is: **Do NOT make a single Large Language Model (or Vision-Language Model) responsible for the entire surveillance pipeline.**

### 1. Real-Time Latency vs. Throughput
* **Specialized CV Pipeline**: YOLO, ByteTrack, and OSNet run at **30 to 60+ FPS** on standard GPUs. Processing 3 multi-camera feeds concurrently requires under 33 ms per frame.
* **Monolithic Multimodal LLMs**: Answering simple visual questions on high-resolution video frames with models like GPT-4V or large VLMs takes **800 ms to 3+ seconds per frame**. Real-time video processing is computationally impossible with an LLM at 30 FPS.

### 2. Bounding Box & Coordinate Precision
* Specialized object detectors possess dedicated regression heads trained explicitly to output exact pixel coordinates `[x1, y1, x2, y2]`.
* Generative VLMs predict text tokens; they frequently produce jittery, imprecise bounding box coordinates and struggle with continuous multi-object tracking.

### 3. Hardware Feasibility & Memory Footprint (RTX 3050 4GB VRAM)
* Specialized models are lightweight:
  * `yolov8n`: ~6 MB VRAM footprint
  * `OSNet-x0.25`: ~10 MB VRAM footprint
  * `PaddleOCR-mobile`: ~15 MB VRAM footprint
* Running this entire specialized pipeline fits comfortably within **2–3 GB VRAM**, making local deployment on consumer hardware (e.g., RTX 3050 Laptop GPU) completely feasible.
* In contrast, even a small 7B VLM requires 6–14 GB VRAM just to load weights, causing instant Out-Of-Memory (OOM) crashes on edge workstations.

### 4. Determinism vs. Hallucination
* For legal and municipal enforcement, traffic alerts must be 100% deterministic and auditable.
* Dedicated OCR models report per-character confidence scores. Generative LLMs are prone to hallucinating characters or misreading digits when unconstrained.

### 5. Modular Debuggability & Maintainability
* If a license plate is misread, the engineer tunes the OCR binarization or font model without touching the tracker.
* If tracking fails during occlusion, the Kalman filter and motion parameters are adjusted.
* In a monolithic LLM, failures are black-box events that cannot be surgically diagnosed or tuned.

### The True Role of the VLM/LLM
The VLM is deployed as a **high-level cognitive auditor**. It does **not** process raw frames in the real-time loop. Instead, when the CV pipeline flags an anomaly (e.g., suspected hit-and-run, disputable parking violation, or complex accident), the system clips a 5-second video buffer and passes it to the VLM to generate an incident summary report in natural language.

---

## 5. Development Phases

We build this system step-by-step across 10 progressive phases:

* **Phase 1: Project Scaffolding & Environment Setup (Current)**
  * Establish clean repository structure.
  * Verify local hardware (CUDA, GPU, Python, libraries).
  * Configure environment verification script.
* **Phase 2: Vehicle & License Plate Detection**
  * Implement YOLO-based vehicle localization (cars, buses, trucks, motorcycles).
  * Implement secondary micro-detector for license plate extraction.
* **Phase 3: High-Accuracy License Plate OCR**
  * Integrate PaddleOCR for plate character extraction.
  * Implement image enhancement (grayscale, thresholding, perspective warp).
  * Add regex validation for Indian vehicle registration formats.
* **Phase 4: Single-Camera Multi-Vehicle Tracking**
  * Integrate ByteTrack / BoT-SORT algorithms.
  * Assign and maintain intra-camera Track IDs across occlusions and motion blur.
* **Phase 5: Deep Appearance Feature Extraction & Vehicle Re-ID**
  * Integrate OSNet (Omni-Scale Network).
  * Extract 512-dimensional normalized visual feature vectors from vehicle crops.
* **Phase 6: Cross-Camera Spatio-Temporal Association & Trajectory Reconstruction**
  * Build the multi-camera matching engine combining cosine visual similarity, OCR string distance, and camera-to-camera transit windows.
  * Reconstruct global continuous vehicle trajectories across multiple cameras.
* **Phase 7: Traffic Event Analysis & Violation Detection**
  * Calculate vehicle speeds, detect wrong-way movement, illegal turns, and red-light crossings.
* **Phase 8: Optional VLM Cognitive Reasoning**
  * Integrate Qwen3-VL to analyze flagged incident clips and answer natural language queries.
* **Phase 9: FastAPI Backend & Real-time Telemetry Service**
  * Build REST endpoints for search, track queries, and video streams.
  * Implement WebSocket broadcast for real-time traffic alerts.
* **Phase 10: System Integration, Benchmarking & Dashboard Connection**
  * Comprehensive end-to-end evaluation (mAP, MOTA, IDF1, Re-ID Rank-1).
  * Connect the AI engine to the web dashboard frontend.

---

## 6. Directory Structure

```
SIH2026-AI-ENGINE/
├── README.md               # Master architecture documentation & project guide
├── requirements.txt        # Python dependency manifest (staged per phase)
├── .gitignore              # Ignores weights, datasets, virtual environments, and caches
├── backend/                # FastAPI application, REST endpoints, WebSocket feeds, schemas
├── ai_engine/              # Core pipeline orchestrator linking all modular stages
├── models/                 # Model weights (.pt, .pth, .onnx, .engine) - excluded from git
├── datasets/               # Sample videos, calibration frames, and benchmark datasets
├── training/               # Custom training and fine-tuning scripts for YOLO and Re-ID
├── inference/              # Standalone video and RTSP multi-stream inference runners
├── tracking/               # Single-camera tracker wrappers (ByteTrack, BoT-SORT)
├── ocr/                    # License plate detection, rectification, and PaddleOCR engine
├── reid/                   # OSNet feature extraction and vehicle re-identification modules
├── trajectory/             # Cross-camera matcher and spatio-temporal route reconstruction
├── evaluation/             # Metrics calculation (mAP, MOTA, IDF1, Rank-1 accuracy)
├── configs/                # YAML configuration files for pipeline and camera settings
├── scripts/                # Utility and diagnostic scripts (e.g., check_environment.py)
└── tests/                  # Automated unit and integration tests (pytest)
```

---

## 7. Planned Local Deployment Architecture

For testing and demonstration, the system runs locally on standard workstation hardware:

```
+--------------------------------------------------------------------------+
|                       LOCAL HOST WORKSTATION                             |
|  OS: Windows 10/11 x64  |  GPU: NVIDIA RTX 3050 (4GB VRAM)  |  RAM: 16GB  |
+--------------------------------------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
+-----------------------------+             +------------------------------+
| AI ENGINE INFERENCE THREAD  |             | FASTAPI ASYNC BACKEND        |
| - OpenCV Frame Ingestion    |   Queue     | - REST API Endpoints         |
| - GPU Tensor Pipeline (FP16)| ----------> | - WebSocket Live Streamer    |
| - CUDA Multi-Model Serving  |             | - SQLite / In-Memory DB      |
+-----------------------------+             +------------------------------+
                                                           |
                                                           v
                                            +------------------------------+
                                            | FUTURE WEB DASHBOARD (UI)    |
                                            | - Multi-camera grid view     |
                                            | - Live trajectory maps       |
                                            | - ANPR & Violation alerts    |
                                            +------------------------------+
```

---

## 8. Phase 1 Setup & Getting Started

### Step 1: Clone or Navigate to the Repository
Open a terminal (PowerShell or Command Prompt) and change into the project directory:
```powershell
cd "d:\apps\coding\hackathon projects\SIH 2026\SIH2026-AI-ENGINE"
```

### Step 2: Create and Activate a Python Virtual Environment
Using Python 3.10 or 3.11:
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (If PowerShell displays execution policy error, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
```

### Step 3: Install PyTorch with CUDA Support
For NVIDIA GPUs (e.g., RTX 3050), install CUDA-enabled PyTorch:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Remaining Stage 1 Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Verify the Environment
Execute the diagnostic script:
```powershell
python scripts/check_environment.py
```

When all dependencies and GPU acceleration are verified, you are ready to proceed with **Phase 2**!

---

## 9. Phase 2: Pretrained YOLO Inference

In Phase 2, we establish our first working perception baseline: running accelerated local object detection on static images and video feeds using **Ultralytics YOLO** and **CUDA GPU acceleration**.

### Key Machine Learning Concepts (Beginner's Guide)

#### 1. What is a Pretrained Model?
* When training a deep neural network from scratch, all mathematical weights (connections between artificial neurons) start with random numbers. Teaching the network requires millions of labeled images and days or weeks of compute time.
* A **pretrained model** (such as `yolov8n.pt`) is a neural network whose weights have already been optimized by training on a massive benchmark dataset—specifically the **COCO (Common Objects in Context)** dataset containing 80 common object categories.
* Using pretrained weights allows us to instantly detect objects out-of-the-box without training or expensive compute.

#### 2. What is Inference?
* **Training** is the process where a model looks at data, calculates its errors (loss), and adjusts its internal weights through backpropagation.
* **Inference** is the forward pass: running new, unseen images through the frozen model to make predictions. No weights are modified during inference; the model simply processes pixel values through convolutional layers and outputs predicted coordinates and labels.

#### 3. What is YOLO (You Only Look Once)?
* Traditional computer vision detectors (like R-CNN or Faster R-CNN) used two stages: first proposing candidate regions, then classifying each region separately. This was accurate but slow (often < 10 FPS).
* **YOLO** revolutionizes this by reframing detection as a single regression problem. In a single forward pass, the entire image is evaluated at once by dividing it into a feature grid and predicting bounding boxes and class probabilities simultaneously.
* This single-stage architecture enables real-time speeds: over **100+ FPS** on modern GPUs.

#### 4. Bounding Boxes
* A bounding box is a rectangular boundary drawn around a detected object.
* Ultralytics YOLO represents bounding boxes in pixel coordinates format `[x1, y1, x2, y2]`:
  * `(x1, y1)`: Top-left corner coordinates of the box.
  * `(x2, y2)`: Bottom-right corner coordinates of the box.

#### 5. Confidence Score
* For every detected bounding box, the model outputs a **confidence score** between `0.0` and `1.0` (or 0% to 100%).
* It represents the mathematical probability that an object actually exists inside the box AND belongs to the predicted category.
* Setting a confidence threshold (e.g., `--conf 0.25`) filters out noisy or uncertain predictions.

#### 6. Class IDs and COCO Mapping
* Neural networks do not directly output English words; they output integer indexes known as **Class IDs**.
* In the standard 80-class COCO taxonomy:
  * `Class 2`: `car`
  * `Class 3`: `motorcycle`
  * `Class 5`: `bus`
  * `Class 7`: `truck`
* A lookup table (`model.names`) maps these numeric class IDs into readable strings.

#### 7. CUDA Hardware Acceleration & 4 GB VRAM Management
* Computer vision models perform billions of matrix multiplications per frame. Running these calculations on a general-purpose CPU takes several hundred milliseconds per frame.
* **NVIDIA CUDA** enables PyTorch and YOLO to offload tensor mathematics to the parallel cores of your dedicated NVIDIA RTX 3050 Laptop GPU, cutting inference latency down to under **10 milliseconds** (~100+ FPS).
* **Memory Safety**: The lightweight Nano model (`yolov8n.pt`) requires only **~6.2 MB of storage** and consumes **~300 MB of VRAM**, safely operating well below the 4 GB hardware ceiling without risking Out-Of-Memory (OOM) crashes.

---

### Why is this Pretrained Model Only a Baseline?

The generic `yolov8n.pt` model is trained on standard international imagery. While it excels at identifying high-level vehicles (`car`, `motorcycle`, `bus`, `truck`), it has critical architectural limitations for our final goal:
1. **No License Plate Class**: The generic COCO dataset has no category for number plates.
2. **Cannot Read Characters**: Object detection localizes boxes; it does not perform optical character recognition (OCR).
3. **No Indian Traffic Specialization**: It lacks annotations for Indian-specific vehicle categories such as auto-rickshaws, e-rickshaws, customized tempos, and two-line motorcycle plates.

### Why Do We Need Custom Training Later for Indian License Plates?

1. **Extreme Scale Difference**: A vehicle fills hundreds of pixels, while a license plate may only be 40x15 pixels in a high-angle surveillance camera. A dedicated secondary micro-detector is required.
2. **Indian License Plate Diversity**:
   * White plates with black text (private vehicles).
   * Yellow plates with black text (commercial taxis, trucks, buses).
   * Green plates with white text (electric vehicles).
   * Black plates with yellow text (self-drive rental vehicles).
   * Military plates featuring upward-pointing arrows.
3. **Multi-line Plate Geometries**: Two-wheelers and auto-rickshaws in India frequently mount stacked two-line plates (e.g., state code on top, digits on the bottom).
4. **Harsh Visual Realities**: High occlusion, non-standard decorative fonts, dust, and varied night illumination necessitate fine-tuning on regional Indian traffic datasets.

---

### Phase 2 Inference Scripts & Usage

#### 1. Image Inference (`inference/image_inference.py`)

Run vehicle detection on a test image:

```powershell
# Run with default sample image (data/images/traffic_test.jpg)
python inference/image_inference.py --source data/images/traffic_test.jpg

# Optional: adjust confidence threshold (e.g. 0.40)
python inference/image_inference.py --source data/images/traffic_test.jpg --conf 0.40

# Optional: detect all COCO classes (including pedestrians, traffic lights, etc.)
python inference/image_inference.py --source data/images/traffic_test.jpg --all-classes
```

**Outputs generated:**
* Annotated image with bounding boxes, labels, and confidences saved to: `data/outputs/traffic_test_annotated.jpg`
* Detailed console breakdown of detected vehicles, confidence scores, and GPU inference timings.

#### 2. Video Inference (`inference/video_inference.py`)

Run frame-by-frame vehicle detection on a video file:

```powershell
# Run on sample video
python inference/video_inference.py --source data/videos/test.mp4

# Optional: process only the first 100 frames for a quick benchmark
python inference/video_inference.py --source data/videos/test.mp4 --max-frames 100
```

**Outputs generated:**
* Annotated MP4 video saved to: `data/outputs/test_annotated.mp4`
* Real-time console progress with current frame, percentage, and FPS.

---

### Phase 2.2 Baseline Experiment: Confidence Thresholding

In Phase 2.2, we study how the **confidence threshold (`--conf`)** parameter directly governs the trade-off between **detection sensitivity (Recall)** and **detection certainty (Precision)**.

#### 1. What is Confidence Thresholding?
When YOLO evaluates an image, its prediction heads generate thousands of candidate bounding boxes across feature grids. For each candidate box, the network computes:
$$\text{Score} = P(\text{Object}) \times P(\text{Class} \mid \text{Object})$$
The **confidence threshold ($\tau$)** acts as a hard mathematical filter applied during post-processing. Any candidate detection whose score is below $\tau$ is immediately rejected.

#### 2. The Precision vs. Recall Trade-off

```
                     CONFIDENCE THRESHOLD SPECTRUM
   0.10             0.25 (Default)         0.50             0.80
<---|---------------------|-----------------|-----------------|--->
  HIGH RECALL                                          HIGH PRECISION
  - Detects distant/small objects                      - Extremely certain detections only
  - Higher risk of false alarms                        - High risk of missed vehicles (false negatives)
```

* **Low Threshold (`--conf 0.25`) &rarr; High Recall:**
  * Catches faint, partially occluded, distant, or angled vehicles.
  * In `traffic_test2.jpg`, this detected **29 vehicles** (including 4 distant trucks and 6 motorcycles).
  * Ideal when downstream modules (e.g. ByteTrack) have secondary validation mechanisms.
* **Medium Threshold (`--conf 0.40 - 0.50`) &rarr; Balanced Baseline:**
  * Eliminates borderline or ambiguous shapes while preserving clear vehicles.
  * Detections scaled down cleanly from **15 to 9 vehicles**.
* **High Threshold (`--conf 0.60+`) &rarr; High Precision:**
  * Only vehicles with unambiguous visual features are retained (**7 detections**).
  * Distant vehicles, partially occluded cars, and smaller two-wheelers are dropped (False Negatives).

#### 3. Impact on Downstream Surveillance Stages
* **For Tracking (Phase 4):** Setting $\tau$ too high causes a car to "disappear" for a few frames under shadows or occlusions, which terminates its single-camera Track ID and generates a fragmented trajectory.
* **For License Plate Detection (Phase 3):** We crop bounding boxes of detected vehicles to find their license plates. If a vehicle is missed due to an overly aggressive confidence threshold, its license plate can never be read.
* **Recommended Default:** In dense traffic surveillance, `--conf 0.25` to `--conf 0.35` is standard practice.

#### 4. Empirical Benchmark Comparison (`traffic_test2.jpg`)

| Confidence Flag | Total Detections | Breakdown by Vehicle Class | Saved Output Artifact |
| :--- | :---: | :--- | :--- |
| `--conf 0.25` | **29** | car: 15, motorcycle: 6, bus: 4, truck: 4 | `data/outputs/traffic_test2_conf0.25_annotated.jpg` |
| `--conf 0.40` (Selected Baseline) | **15** | car: 8, motorcycle: 4, bus: 3 | `data/outputs/traffic_test2_conf0.40_annotated.jpg` |
| `--conf 0.50` | **9** | car: 6, bus: 2, motorcycle: 1 | `data/outputs/traffic_test2_conf0.50_annotated.jpg` |
| `--conf 0.60` | **7** | car: 4, bus: 2, motorcycle: 1 | `data/outputs/traffic_test2_conf0.60_annotated.jpg` |

Each execution generates a unique, non-overwriting file in `data/outputs/` tagged with its confidence value.

#### 5. Baseline Operating Threshold Selection

> [!IMPORTANT]
> **Current Baseline Operating Threshold: `0.40`**
>
> * **Current Role**: `--conf 0.40` is formally recorded as our **interim baseline operating threshold** for ongoing pipeline development. It eliminates ambiguous edge-border artifacts and low-confidence background boxes while maintaining solid recall for prominent vehicles (cars, buses, motorcycles).
> * **Not the Final SIH Threshold**: This is strictly an exploratory baseline on the generic pretrained COCO model. It is **NOT** the final operating threshold for the SIH 2026 surveillance engine.
> * **Future Validation Protocol**: The definitive threshold will be systematically re-evaluated and calibrated using **Precision-Recall (PR) curves, F1-score maximization, and mAP@0.5:0.95 validation** once we integrate custom Indian traffic datasets and ground-truth bounding box annotations.
