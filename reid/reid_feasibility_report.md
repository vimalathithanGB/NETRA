# PHASE 5B: Vehicle Re-Identification (Re-ID) Environment & Feasibility Diagnostic

**Date:** 2026-09-18  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Host Environment:** Windows, Python 3.11.5, PyTorch 2.6.0+cu124 (CUDA active)

---

## 1. Environment & Library Inventory

| Package / Library | Status | Version | Notes |
|---|---|---|---|
| **Python** | Installed | `3.11.5` | Standard 64-bit CPython |
| **PyTorch (`torch`)** | Installed | `2.6.0+cu124` | CUDA active on RTX 3050 |
| **`torchvision`** | Installed | `0.21.0+cu124` | CUDA enabled |
| **`ultralytics`** | Installed | `8.4.152` | YOLOv8n detector & tracker backend |
| **`huggingface_hub`** | Installed | `1.31.0` | Accessible for remote weights fetching |
| **`opencv-python`** | Installed | `5.0.0.93` | Image manipulation & video processing |
| **`torchreid`** | **Not Installed** | N/A | Older library; often conflicts with PyTorch 2.6 on Win |
| **`timm`** | **Not Installed** | N/A | Vision models collection |
| **`transformers`** | **Not Installed** | N/A | Hugging Face transformer ecosystem |
| **`boxmot`** | **Not Installed** | N/A | Multi-object tracking & Re-ID toolkit |

---

## 2. Re-ID Taxonomy: Critical Functional Distinctions

To ensure sound architectural choices, three distinct model categories must be clearly delineated:

### A. Generic Image Embeddings (e.g. ImageNet Backbones)
* **Backbones:** ResNet18, ResNet50, MobileNetV3 (from `torchvision.models`).
* **Objective:** Categorize objects into 1,000 distinct semantic classes (e.g., distinguishing a bus from a bicycle).
* **Vehicle Re-ID Suitability:** **Poor / Inadequate**. ImageNet backbones collapse intra-class visual differences. They cannot reliably distinguish two silver sedans of the same make/model.

### B. Person Re-ID Models (e.g. Market-1501, MSMT17, DukeMTMC)
* **Backbones:** OSNet x1.0 / x0.25 (`osnet_x1_0_msmt17.pth`, `osnet_x0_25_market1501.pt`).
* **Objective:** Extract vertical human silhouette features (aspect ratio ~2:1, clothing colors, bags, shoes).
* **Vehicle Re-ID Suitability:** **Unsuitable for production vehicle identification**. While the architecture is powerful, the pre-trained feature space looks for torso/leg proportions rather than horizontal vehicular landmarks (grilles, windshield decals, roof racks, wheel hubs, taillights).

### C. True Vehicle Re-ID Models (e.g. VeRi-776, VehicleID, CityFlow)
* **Backbones:** OSNet-AIN, ResNet34/50-ReID, CLIP-VehicleID.
* **Objective:** Explicit metric learning on vehicles across multiple disjoint camera perspectives under varied angles and lighting.
* **Vehicle Re-ID Suitability:** **Native & High-Precision**. Specifically trained to isolate fine-grained visual identity markers of vehicles.

---

## 3. Pretrained Candidate Evaluation for RTX 3050 (4 GB VRAM)

| Candidate Model | Training Dataset | Model Size | Parameter Count | Est. VRAM (Inference) | Suitability for Vehicle Re-ID | Hardware Viability |
|---|---|---|---|---|---|---|
| **OSNet-AIN x1.0 (`vehicle-reid-0001`)** | **VeRi-776** | **~8.8 MB** | **2.19 M** | **~150 - 250 MB** | **Native Vehicle Re-ID** (Rank-1 96.3%, mAP 85.1%) | **Optimal (Fast, ultra-low memory)** |
| **ResNet-34 ReID (`dgwon/resnet-34-veri776`)** | **VeRi-776** | **~85 MB** | **21.3 M** | **~350 - 500 MB** | **Native Vehicle Re-ID** (Rank-1 ~94%) | **Good (Higher compute than OSNet)** |
| **CLIP-VehicleID (`clip_vehicleid.pt`)** | **VehicleID** | **~350 - 600 MB** | **86 M+** | **~1.5 - 2.2 GB** | **Native Vehicle Re-ID** | **Marginal (Risks VRAM pressure with YOLOv8n)** |
| **OSNet x0.25 (MSMT17)** | *MSMT17 (Person)* | ~2.5 MB | 0.5 M | ~80 MB | *Person Re-ID Only* (Negative match for vehicles) | Incompatible Domain |
| **MobileNetV3 / ResNet50** | *ImageNet-1k* | ~12 - 98 MB | 5 - 25 M | ~200 - 400 MB | *Generic Classification* (Low discrimination) | Incompatible Task |

---

## 4. Key Takeaways & Recommendations

1. **Architecture Recommendation:** **OSNet-AIN x1.0** trained on **VeRi-776** (derived from the Intel Open Model Zoo `vehicle-reid-0001` specification) is the best match.
   - It requires only **~8.8 MB** storage and **< 250 MB VRAM**, leaving more than 3.5 GB VRAM for YOLOv8n, ByteTrack, and video frame buffering.
   - It is specifically tuned for vehicle re-identification across camera angles.
2. **Next Step:** 
   - Construct a lightweight standalone PyTorch extractor module in `reid/osnet_reid.py` that can load these weights without needing bulky external tracking libraries.
   - Test embedding extraction and cosine similarity on cropped vehicle bounding boxes from `data/videos/test_2.mp4`.
