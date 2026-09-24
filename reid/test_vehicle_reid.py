"""
SIH 2026 NETRA AI ENGINE - Vehicle Re-ID Verification Test
==========================================================
Script: reid/test_vehicle_reid.py

Description:
    Validates the standalone Vehicle Re-Identification (Re-ID) module:
    1. Loads the OSNet-AIN x1.0 pretrained model onto the active device (CUDA/CPU).
    2. Runs inference on a test vehicle crop (OpenCV BGR format).
    3. Verifies embedding properties (shape, dtype, finite values, unit L2 norm).
    4. Computes self-cosine similarity (must be approximately 1.0).
    5. Tests batch extraction and pairwise comparison.
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.vehicle_reid import VehicleReIDExtractor, compute_cosine_similarity


def run_verification():
    print("=" * 70)
    print("SIH 2026 NETRA - VEHICLE RE-IDENTIFICATION VERIFICATION TEST")
    print("=" * 70)

    # 1. Initialize Extractor
    extractor = VehicleReIDExtractor()
    info = extractor.get_model_info()

    # 2. Create a test vehicle crop (e.g. simulated car crop: 160 x 240 BGR)
    # Uses a realistic colored synthetic crop with vehicle-like structure
    np.random.seed(42)
    synthetic_crop = np.zeros((180, 260, 3), dtype=np.uint8)
    synthetic_crop[:, :] = (180, 50, 40)  # Metallic blue vehicle body
    # Simulated windshield / roof
    synthetic_crop[20:70, 40:220] = (70, 70, 70)
    # Simulated headlights
    synthetic_crop[130:160, 20:60] = (240, 240, 240)
    synthetic_crop[130:160, 200:240] = (240, 240, 240)

    # If video test_2.mp4 exists, grab a real frame crop for realistic visual verification
    test_video_path = PROJECT_ROOT / "data" / "videos" / "test_2.mp4"
    real_crop = None
    if test_video_path.exists():
        cap = cv2.VideoCapture(str(test_video_path))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            # Crop a central vehicle area (e.g. 200x200 patch)
            h, w = frame.shape[:2]
            real_crop = frame[h // 4 : h // 4 + 200, w // 4 : w // 4 + 200].copy()

    test_crop = real_crop if real_crop is not None else synthetic_crop

    # 3. Extract single embedding
    embedding = extractor.extract_embedding(test_crop)
    l2_norm = float(np.linalg.norm(embedding))

    # 4. Print Mandatory Diagnostics
    print(f"Model loaded: {info['model_name']} ({info['trained_dataset']})")
    print(f"Device: {info['device']} ({info['device_name']})")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding L2 norm: {l2_norm:.6f}")
    print("-" * 70)

    # 5. Assertions & Verification
    # (a) Shape verification
    assert embedding.shape == (512,), f"Expected shape (512,), got {embedding.shape}"
    print("[PASS] Shape check: exactly (512,)")

    # (b) Finite values
    assert np.all(np.isfinite(embedding)), "Embedding contains NaN or Inf values!"
    print("[PASS] Numerical check: all 512 values are finite")

    # (c) L2 norm check
    assert abs(l2_norm - 1.0) < 1e-4, f"L2 norm {l2_norm:.6f} is not approximately 1.0!"
    print(f"[PASS] Normalization check: L2 norm is {l2_norm:.6f} (~1.0)")

    # 6. Self-Cosine Similarity Check
    self_sim = compute_cosine_similarity(embedding, embedding)
    print(f"Self cosine similarity: {self_sim:.6f}")
    assert abs(self_sim - 1.0) < 1e-4, f"Self-similarity {self_sim:.6f} is not 1.0!"
    print(f"[PASS] Self-similarity check: {self_sim:.6f} (~1.0)")

    # 7. Distinct Crop Discriminative Verification
    # Compare with a different crop (e.g. modified hue / synthetic crop)
    modified_crop = cv2.bitwise_not(test_crop)
    diff_embedding = extractor.extract_embedding(modified_crop)
    diff_sim = compute_cosine_similarity(embedding, diff_embedding)
    print(f"Distinct vehicle comparison similarity: {diff_sim:.4f}")
    assert diff_sim < 0.95, f"Different crops should have distinct embeddings, got sim: {diff_sim}"
    print("[PASS] Discrimination check: distinct vehicles yield contrasting embeddings")

    # 8. Batch Extraction Verification
    batch_embeddings = extractor.extract_batch_embeddings([test_crop, modified_crop])
    assert batch_embeddings.shape == (2, 512), f"Batch shape error: {batch_embeddings.shape}"
    batch_norms = np.linalg.norm(batch_embeddings, axis=1)
    assert np.allclose(batch_norms, 1.0, atol=1e-4), "Batch embeddings not L2 normalized!"
    print(f"[PASS] Batch extraction check: shape {batch_embeddings.shape}, all L2 norms ~ 1.0")

    print("=" * 70)
    print("[ALL TESTS PASSED SUCCESSFULLY] Vehicle Re-ID module is operational.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
