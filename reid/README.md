# SIH 2026 NETRA — Vehicle Re-Identification (Vehicle Re-ID) Module

## 1. What is Vehicle Re-Identification (Re-ID)?

**Vehicle Re-Identification (Vehicle Re-ID)** is the computer vision task of recognizing and associating the identity of a specific vehicle across different non-overlapping camera viewpoints, across varying times, and under diverse environmental conditions (lighting, angles, occlusions, and distances).

While standard vehicle classification assigns generic category labels (e.g., *car*, *truck*, *auto_rickshaw*), Vehicle Re-ID extracts a unique, high-dimensional **visual appearance fingerprint** (an embedding vector) that allows the system to determine whether a vehicle detected by Camera 1 at intersection A is the exact same physical vehicle observed minutes later by Camera 2 at intersection B.

```
┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────────┐
│  Vehicle Crop   │  ──>  │  OSNet-AIN Re-ID Model │  ──>  │  512-Dim Feature Vector   │
│  (OpenCV BGR)   │       │  (VeRi-776 Pretrained) │       │  (L2-Normalized Embedding)│
└─────────────────┘       └────────────────────────┘       └───────────────────────────┘
                                                                         │
                                                                         ▼
                                                          Cosine Similarity Comparison
                                                          sim(u, v) = u · v ∈ [-1, 1]
```

---

## 2. Why Vehicle-Specific Re-ID (vs Person Re-ID or Generic Embeddings)?

A critical architectural distinction in deep metric learning is domain specialization:

1. **Generic ImageNet Embeddings (e.g. standard ResNet/MobileNet):**
   * *Objective:* Coarse inter-class classification (distinguishing a bus from a bird).
   * *Limitation:* Compresses intra-class variance. Cannot reliably distinguish two white sedans of identical make.
2. **Person Re-ID Models (e.g. Market-1501, MSMT17):**
   * *Objective:* Tall vertical human silhouettes (~2:1 aspect ratio) focused on clothing color, bags, legs, and facial structures.
   * *Limitation:* Totally inappropriate geometric priors and feature representations for vehicles.
3. **Vehicle-Specific Re-ID (e.g. VeRi-776, VehicleID):**
   * *Objective:* Horizontal vehicular geometry (~1:1 or 4:3) specifically trained to focus on fine-grained identity markers: grille shapes, headlights, windshield decals, roof racks, body dents, window tints, and tail fin styling.
   * *NETRA Choice:* **OSNet-AIN x1.0 trained on VeRi-776** (`vehicle-reid-0001` architecture).

---

## 3. ByteTrack vs Vehicle Re-ID: Key Differences

| Feature | ByteTrack (Phase 5A) | Vehicle Re-ID (Phase 5B) |
|---|---|---|
| **Scope** | Single-camera, continuous video stream | Cross-camera, discontinuous views |
| **Primary Signals** | Spatial motion, bounding box overlap (IoU), Kalman filter | Deep visual appearance embeddings |
| **Temporal Horizon** | Short-term (frames / seconds within same camera) | Long-term (minutes / hours across multiple cameras) |
| **Occlusion Recovery** | Recovers from brief occlusions (e.g., 30 frames) | Recovers identity even after vehicle leaves view completely |
| **Model** | YOLOv8n detector + ByteTrack association | OSNet-AIN x1.0 metric extractor |

---

## 4. Input & Output Specification

* **Input Image / Crop:**
  * Format: OpenCV / NumPy BGR image (`numpy.ndarray`).
  * Channels: 3 (BGR order).
  * Data Type: `uint8` (`[0, 255]`).
  * Resolution: Any crop size (preprocessed internally to $208 \times 208$).
* **Internal Preprocessing:**
  * BGR to RGB color conversion.
  * Bilinear resizing to $208 \times 208$.
  * Scaling to $[0.0, 1.0]$ float32.
  * Dynamic channel standardization via `InstanceNorm2d`.
* **Output Embedding:**
  * Shape: `(512,)` for single crop; `(N, 512)` for batch crops.
  * Data Type: `numpy.float32`.
  * Normalization: **Strictly L2-normalized** ($\|e\|_2 \approx 1.0$).

---

## 5. Cosine Similarity Metric

Because all embeddings produced by the module are strictly L2-normalized to unit Euclidean length ($\|u\|_2 = 1, \|v\|_2 = 1$), the cosine similarity reduces to a fast dot product:

$$\text{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \sum_{i=1}^{512} u_i v_i$$

* **1.0**: Identical visual appearance (e.g., comparing an embedding with itself).
* **> 0.75**: High probability of the same physical vehicle identity.
* **0.40 – 0.75**: Moderate visual similarity (same make/color/class).
* **< 0.40**: Dissimilar / different vehicles.

---

## 6. CPU / GPU Execution Behavior

* **Default Behavior:** Automatically selects NVIDIA CUDA acceleration (`cuda:0`) if PyTorch detects a CUDA-capable GPU.
* **Target Hardware (RTX 3050 Laptop GPU, 4 GB VRAM):**
  * Single crop latency: $\approx 25-40\text{ ms}$.
  * Batch throughput ($B=8$): $> 200\text{ crops/sec}$.
  * VRAM consumption: $\approx 180-250\text{ MB}$ (leaves plenty of headroom for YOLOv8n).
* **Fallback Behavior:** If CUDA is unavailable or explicitly disabled, executes gracefully on CPU ($\approx 18-25\text{ ms}$ latency via OpenMP threads) with zero code changes required.

---

## 7. Model Weights Location & Verification

The weights are stored in the root `weights/` directory:

| File | Size | Description |
|---|---|---|
| `weights/osnet_ain_x1_0_vehicle_reid.pt` | ~8.8 MB | Native PyTorch `state_dict` (559 parameters) |
| `weights/osnet_ain_x1_0_vehicle_reid.onnx` | ~8.8 MB | Source ONNX model (OpenVINO Open Model Zoo) |

* **Automatic Setup:** If the weights are absent, `VehicleReIDExtractor` automatically downloads the verified ONNX model from the official Intel Open Model Zoo repository and validates its SHA-384 checksum:
  `0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7`
* **License:** MIT License (Kaiyang Zhou / Intel Corporation).

---

## 8. Integration Roadmap: Connecting to ByteTrack

In subsequent milestones, this standalone module will interface with the tracking pipeline:

```
Video Frame (CCTV 1)
         ↓
YOLOv8n Vehicle Detector
         ↓
Bounding Box Crops [car_1, car_2, ...]
         ↓
ByteTrack (Single-Camera Spatial Tracking)
         ↓  (At periodic intervals / high-confidence frames)
VehicleReIDExtractor.extract_batch_embeddings()
         ↓
Feature Gallery / Cross-Camera Matching Engine
         ↓
Global Trajectory & Re-ID Across CCTV 1, 2, ... N
```

By keeping the feature extractor strictly decoupled, it can be tested, benchmarked, and upgraded independently of detection and tracking algorithms.
