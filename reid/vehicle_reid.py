"""
SIH 2026 NETRA AI ENGINE - Vehicle Re-Identification (Vehicle Re-ID) Module
===========================================================================
Module: reid/vehicle_reid.py

Description:
    Standalone Vehicle Re-Identification (Re-ID) feature extractor.
    Extracts deep, 512-dimensional visual appearance embeddings from cropped
    vehicle bounding boxes using the OSNet-AIN x1.0 architecture pretrained
    specifically for vehicle re-identification on the VeRi-776 surveillance dataset.

Key Design Principles:
    1. Complete Decoupling: Fully independent of YOLO, ByteTrack, FastAPI, DB, and UI.
    2. Zero Heavy Third-Party Dependencies: Uses pure PyTorch (no torchreid/boxmot/timm).
    3. Hardware Optimized: Automatically leverages NVIDIA CUDA GPU (RTX 3050 4GB)
       with seamless fallback to CPU if CUDA is unavailable.
    4. Robust Error Handling: Explicit validation for missing weights, corrupt/empty
       crops, device fallbacks, and output dimensionality.
    5. Unit Normalized: Embeddings are strictly L2-normalized so that cosine similarity
       equals the dot product.

Usage Example:
    ```python
    import cv2
    from reid.vehicle_reid import VehicleReIDExtractor, compute_cosine_similarity

    # 1. Initialize extractor (auto-selects CUDA if available)
    extractor = VehicleReIDExtractor()

    # 2. Extract embedding from a cropped vehicle (OpenCV BGR format)
    crop = cv2.imread("sample_car.jpg")
    embedding = extractor.extract_embedding(crop)
    # embedding.shape -> (512,)

    # 3. Compare similarity between two vehicle crops
    similarity = compute_cosine_similarity(embedding_1, embedding_2)
    # similarity -> float in range [-1.0, 1.0]
    ```
"""

import os
import urllib.request
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reid.osnet_ain import build_osnet_ain_x1_0, OSNetAIN


# =============================================================================
# 0. CONFIGURATION & CONSTANTS
# =============================================================================

DEFAULT_WEIGHTS_PATH = Path("weights/osnet_ain_x1_0_vehicle_reid.pt")
DEFAULT_ONNX_PATH = Path("weights/osnet_ain_x1_0_vehicle_reid.onnx")

# Official OpenVINO Open Model Zoo distribution for vehicle-reid-0001
OMZ_ONNX_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx"
)
EXPECTED_SHA384 = (
    "0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da18916"
    "6343157273c02f3b0fede3050ef7abb7"
)

# Standard input size for OSNet-AIN vehicle re-id model
MODEL_INPUT_HEIGHT = 208
MODEL_INPUT_WIDTH = 208
EMBEDDING_DIM = 512


# =============================================================================
# 1. CUSTOM EXCEPTIONS
# =============================================================================

class ReIDError(Exception):
    """Base exception for Vehicle Re-ID operations."""
    pass


class WeightsNotFoundError(ReIDError):
    """Raised when model weights file cannot be found or downloaded."""
    pass


class InvalidCropError(ReIDError):
    """Raised when an input image/crop is invalid, empty, or unreadable."""
    pass


# =============================================================================
# 2. COSINE SIMILARITY HELPER
# =============================================================================

def compute_cosine_similarity(
    embedding1: Union[np.ndarray, torch.Tensor],
    embedding2: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Computes cosine similarity between two feature embeddings.

    For L2-normalized embeddings, cosine similarity is identical to the inner product:
        cosine_similarity(u, v) = (u . v) / (||u|| * ||v||)

    Args:
        embedding1: First 512-dim embedding (1D numpy array or torch tensor).
        embedding2: Second 512-dim embedding (1D numpy array or torch tensor).

    Returns:
        float in [-1.0, 1.0] representing visual similarity.
    """
    if isinstance(embedding1, torch.Tensor):
        embedding1 = embedding1.detach().cpu().numpy()
    if isinstance(embedding2, torch.Tensor):
        embedding2 = embedding2.detach().cpu().numpy()

    emb1 = np.asarray(embedding1, dtype=np.float32).ravel()
    emb2 = np.asarray(embedding2, dtype=np.float32).ravel()

    if emb1.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"embedding1 must have shape ({EMBEDDING_DIM},), got {emb1.shape}"
        )
    if emb2.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"embedding2 must have shape ({EMBEDDING_DIM},), got {emb2.shape}"
        )

    norm1 = float(np.linalg.norm(emb1))
    norm2 = float(np.linalg.norm(emb2))

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0

    similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
    # Numerical clip to [-1.0, 1.0]
    return max(-1.0, min(1.0, similarity))


# =============================================================================
# 3. WEIGHTS AUTO-DISCOVERY & CONVERSION HELPER
# =============================================================================

def ensure_model_weights(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    onnx_path: Path = DEFAULT_ONNX_PATH,
) -> Path:
    """
    Ensures that the PyTorch OSNet-AIN vehicle Re-ID weights file exists.
    If only the verified ONNX model exists, converts its initializers into a PyTorch .pt file.
    If neither exists, downloads and verifies the official ONNX model from OpenVINO storage.

    Returns:
        Resolved absolute Path to the .pt weights file.
    """
    weights_path = Path(weights_path).resolve()
    onnx_path = Path(onnx_path).resolve()

    # 1. If PyTorch weights already exist, return immediately
    if weights_path.exists() and weights_path.stat().st_size > 1000:
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. If ONNX weights do not exist, download them
    if not onnx_path.exists() or onnx_path.stat().st_size < 1000:
        print(f"[*] Downloading verified OSNet-AIN vehicle Re-ID weights from:\n    {OMZ_ONNX_URL}")
        try:
            urllib.request.urlretrieve(OMZ_ONNX_URL, str(onnx_path))
        except Exception as e:
            raise WeightsNotFoundError(
                f"Failed to download vehicle Re-ID weights from {OMZ_ONNX_URL}.\nError: {e}"
            ) from e

    # 3. Verify SHA-384 checksum of ONNX file
    with open(onnx_path, "rb") as f:
        file_sha = hashlib.sha384(f.read()).hexdigest()

    if file_sha != EXPECTED_SHA384:
        raise WeightsNotFoundError(
            f"Checksum verification failed for {onnx_path}!\n"
            f"Expected: {EXPECTED_SHA384}\n"
            f"Received: {file_sha}"
        )

    # 4. Extract initializers to PyTorch state_dict
    try:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        state_dict = {}
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            state_dict[init.name] = torch.from_numpy(arr.copy())

        torch.save(state_dict, str(weights_path))
        print(f"[OK] Successfully prepared PyTorch Re-ID weights: {weights_path}")
        return weights_path
    except Exception as e:
        raise WeightsNotFoundError(
            f"Failed to extract PyTorch state_dict from ONNX model: {e}"
        ) from e


# =============================================================================
# 4. STANDALONE VEHICLE RE-ID EXTRACTOR
# =============================================================================

class VehicleReIDExtractor:
    """
    High-performance, standalone Vehicle Re-Identification feature extractor.

    Input:
        OpenCV BGR image crop of a detected vehicle (H x W x 3, uint8).
    Output:
        512-dimensional, L2-normalized numpy embedding (float32).
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        fp16: bool = False,
    ):
        """
        Initializes the Vehicle Re-ID model.

        Args:
            weights_path: Path to PyTorch model weights. Defaults to weights/osnet_ain_x1_0_vehicle_reid.pt.
            device: 'cuda', 'cuda:0', 'cpu', or None (auto-detects CUDA).
            fp16: If True and on CUDA, enables half-precision (fp16) for faster inference.
        """
        # Resolve weights path
        target_weights = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
        resolved_weights = ensure_model_weights(target_weights)

        # Resolve compute device
        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = str(device)
            if self.device_str.startswith("cuda") and not torch.cuda.is_available():
                print(f"[WARN] CUDA requested ('{self.device_str}') but CUDA is unavailable. Falling back to CPU.")
                self.device_str = "cpu"

        self.device = torch.device(self.device_str)
        self.use_fp16 = bool(fp16 and self.device.type == "cuda")

        # Load PyTorch model
        try:
            self.model: OSNetAIN = build_osnet_ain_x1_0(
                weights_path=resolved_weights,
                device=self.device_str,
            )
            if self.use_fp16:
                self.model.half()
        except Exception as e:
            raise ReIDError(f"Failed to initialize OSNet-AIN vehicle model from {resolved_weights}: {e}") from e

        self.input_size = (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)  # (208, 208)

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a raw OpenCV vehicle crop for OSNet-AIN input.

        Pipeline:
            1. Validates crop dimensions and data type.
            2. Converts BGR -> RGB.
            3. Bilinear resize to (208, 208).
            4. Converts to float32 and scales [0, 255] -> [0.0, 1.0].
            5. Transposes (H, W, C) -> (C, H, W).
            6. Adds batch dimension -> (1, 3, 208, 208).
            7. Sends to target device.

        Args:
            crop: BGR image numpy array.

        Returns:
            Preprocessed PyTorch tensor of shape (1, 3, 208, 208).
        """
        if crop is None:
            raise InvalidCropError("Input crop is None.")

        if not isinstance(crop, np.ndarray):
            raise InvalidCropError(f"Input crop must be a numpy.ndarray, got {type(crop)}.")

        if crop.ndim != 3 or crop.shape[2] != 3:
            raise InvalidCropError(
                f"Input crop must have shape (H, W, 3), got shape {crop.shape}."
            )

        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            raise InvalidCropError(f"Input crop has invalid dimensions: {w}x{h}.")

        # 1. BGR -> RGB
        rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # 2. Resize to (208, 208)
        resized = cv2.resize(
            rgb_img,
            self.input_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # 3. Convert to float32 and scale to [0.0, 1.0]
        # (OSNet-AIN utilizes an initial InstanceNorm2d layer that expects [0, 1] inputs)
        normalized = resized.astype(np.float32) / 255.0

        # 4. Transpose (H, W, C) -> (C, H, W)
        chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))

        # 5. Tensor conversion and batch dimension
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        if self.use_fp16:
            tensor = tensor.half()

        return tensor

    # -------------------------------------------------------------------------
    # Single Embedding Extraction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts a 512-dimensional, L2-normalized feature embedding from a single vehicle crop.

        Args:
            crop: OpenCV BGR vehicle crop (numpy array).

        Returns:
            1D numpy array of shape (512,), dtype float32, with L2 norm ~ 1.0.
        """
        tensor = self.preprocess(crop)

        # Model forward pass
        raw_embedding = self.model(tensor)

        # Strict dimensionality check
        if raw_embedding.shape != (1, EMBEDDING_DIM):
            raise ReIDError(
                f"Model produced unexpected embedding shape: {raw_embedding.shape}. "
                f"Expected: (1, {EMBEDDING_DIM})"
            )

        # L2-normalization (unit Euclidean length)
        normalized_embedding = F.normalize(raw_embedding, p=2, dim=1)

        # Move to host CPU numpy array
        embedding_np = (
            normalized_embedding.squeeze(0)
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        )

        # Check for NaN / Inf
        if not np.all(np.isfinite(embedding_np)):
            raise ReIDError("Extracted embedding contains non-finite values (NaN or Inf).")

        return embedding_np

    # -------------------------------------------------------------------------
    # Batch Embedding Extraction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_batch_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extracts L2-normalized embeddings for a list of vehicle crops in a single batch pass.

        Args:
            crops: List of OpenCV BGR vehicle crops.

        Returns:
            2D numpy array of shape (N, 512), dtype float32.
        """
        if not crops:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        tensors = [self.preprocess(c) for c in crops]
        batch_tensor = torch.cat(tensors, dim=0)

        raw_embeddings = self.model(batch_tensor)
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        embeddings_np = normalized_embeddings.detach().to(torch.float32).cpu().numpy()

        if not np.all(np.isfinite(embeddings_np)):
            raise ReIDError("Extracted batch embeddings contain non-finite values.")

        return embeddings_np

    # -------------------------------------------------------------------------
    # Diagnostics & Metadata
    # -------------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Returns runtime configuration and model metadata."""
        return {
            "model_name": "OSNet-AIN x1.0 (Vehicle Re-ID)",
            "trained_dataset": "VeRi-776",
            "domain": "Vehicle Surveillance Re-Identification",
            "device": self.device_str,
            "device_name": (
                torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
            ),
            "input_resolution": f"{MODEL_INPUT_WIDTH}x{MODEL_INPUT_HEIGHT}",
            "embedding_dim": EMBEDDING_DIM,
            "fp16_enabled": self.use_fp16,
            "weights_path": str(DEFAULT_WEIGHTS_PATH),
        }
