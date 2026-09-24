"""
NETRA — License Plate Optical Character Recognition (OCR) Engine
Component: Model 3 — Plate Text Recognition & Validation
Project: SIH 2026 NETRA

Architecture:
  Plate Crop (OpenCV BGR)
    → Crop Validation
    → Gentle Preprocessing (8% Border Padding, H=48 Aspect-Preserved Resize, LAB-CLAHE)
    → PaddleOCR Recognition-Only (det=False, rec=True, cls=True) on CPU
    → Text Cleaning & Normalization (Uppercase, Alphanumeric Filtering)
    → Position-Aware OCR Ambiguity Correction (O/0, I/1, Z/2, S/5, B/8)
    → Indian Plate Baseline Format Validation (^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$)
    → Structured Output & Fallback OCR against Unmodified Crop
"""

import os
import re
import sys
import logging
from typing import Dict, Any, Tuple, Optional, Set
import cv2
import numpy as np

# Configure module logger
logger = logging.getLogger("NETRA.OCR")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [NETRA.OCR] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Baseline Indian License Plate Regular Expression
# Format: 2-letter State Code + 1-2 digit District Code + 0-3 letter Series Code + 4 digit Registration
# Examples: MH12DE1433, DL1CAB1234, GJ01WC8529, KA531234, TN04AZ6643
# Note: This is a practical baseline rule representing standard civilian, commercial, and transport
# vehicles. It does NOT cover every possible specialized format (such as Bharat Series 22BH...,
# military upward-arrow plates, diplomatic CD/CC series, or temporary numbers).
BASELINE_INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")

# Standard Two-Letter State & Union Territory Codes of India (CMVR Rule 50)
INDIAN_STATE_CODES: Set[str] = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DH", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TR",
    "TS", "UK", "UP", "WB", "OR", "UA"
}

# OCR Character Ambiguity Mapping Tables
# Applied strictly based on expected character position (numeric vs alphabetic)
NUM_TO_ALPHA: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
}

ALPHA_TO_NUM: Dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
}


def clean_plate_text(raw_text: Optional[str]) -> str:
    """
    Cleans raw OCR output text:
    - Converts to uppercase
    - Strips whitespace and linebreaks
    - Removes punctuation and symbols (hyphens, dots, quotes, colons, underscores)
    - Preserves only valid alphanumeric characters [A-Z0-9]
    """
    if not raw_text:
        return ""
    text = str(raw_text).upper().strip()
    return re.sub(r"[^A-Z0-9]", "", text)


def validate_indian_plate_format(text: str) -> bool:
    """
    Validates whether the cleaned registration string strictly matches
    the baseline Indian license plate regular expression:
    ^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$
    """
    if not text:
        return False
    return bool(BASELINE_INDIAN_PLATE_REGEX.match(text))


def disambiguate_indian_plate(text: str) -> Tuple[str, bool]:
    """
    Applies position-aware OCR character disambiguation according to the
    Indian license plate baseline structure.

    Ambiguity substitutions (O<->0, I<->1, Z<->2, S<->5, B<->8) are NEVER applied
    blindly across the entire string. They are only substituted when a character's
    position demands a numeric or alphabetic type:
    - Positions 0 & 1 (State Code): Strictly Alphabetic [A-Z]
    - Position 2 (and 3 if 2-digit RTO): Strictly Numeric [0-9]
    - Middle Series (0-3 chars): Strictly Alphabetic [A-Z]
    - Last 4 Characters (Registration Number): Strictly Numeric [0-9]

    Returns:
        Tuple[str, bool]: (disambiguated_text, is_valid_format)
    """
    clean = clean_plate_text(text)
    if not clean:
        return "", False

    # If already fully valid according to baseline regex, return immediately
    if validate_indian_plate_format(clean):
        return clean, True

    n = len(clean)
    # Valid baseline Indian plate lengths range from 7 (e.g. DL11234) to 11 (e.g. DL10CAB1234)
    if n < 7 or n > 11:
        return clean, False

    best_candidate: Optional[str] = None
    min_penalty = float("inf")

    # Middle characters between state code (first 2) and registration digits (last 4)
    mid_raw = clean[2:n - 4]
    mid_len = len(mid_raw)

    # In India, only Delhi (DL) historically issues 1-digit district codes (DL 1 ... DL 9).
    # All other states strictly use 2-digit district codes (01 to 99).
    state_prefix = clean[:2]
    d_lens = (1, 2) if state_prefix == "DL" else (2, 1)

    for d_len in d_lens:
        s_len = mid_len - d_len
        if s_len not in (0, 1, 2, 3):
            continue

        subs = 0
        cand_parts = []
        possible = True

        # 1. State Code (Indices 0, 1): Expected ALPHABETIC
        for ch in clean[:2]:
            if ch.isalpha():
                cand_parts.append(ch)
            elif ch in NUM_TO_ALPHA:
                cand_parts.append(NUM_TO_ALPHA[ch])
                subs += 1
            else:
                possible = False
                break
        if not possible:
            continue

        # 2. District Code: mid_raw[:d_len] expected NUMERIC
        for ch in mid_raw[:d_len]:
            if ch.isdigit():
                cand_parts.append(ch)
            elif ch in ALPHA_TO_NUM:
                cand_parts.append(ALPHA_TO_NUM[ch])
                subs += 1
            else:
                possible = False
                break
        if not possible:
            continue

        # 3. Series Code: mid_raw[d_len:] expected ALPHABETIC
        for ch in mid_raw[d_len:]:
            if ch.isalpha():
                cand_parts.append(ch)
            elif ch in NUM_TO_ALPHA:
                cand_parts.append(NUM_TO_ALPHA[ch])
                subs += 1
            else:
                possible = False
                break
        if not possible:
            continue

        # 4. Registration Digits (Last 4 chars): Expected NUMERIC
        for ch in clean[-4:]:
            if ch.isdigit():
                cand_parts.append(ch)
            elif ch in ALPHA_TO_NUM:
                cand_parts.append(ALPHA_TO_NUM[ch])
                subs += 1
            else:
                possible = False
                break
        if not possible:
            continue

        candidate_str = "".join(cand_parts)
        if validate_indian_plate_format(candidate_str):
            cand_state = candidate_str[:2]
            # Reward recognized valid Indian state codes
            state_bonus = -0.5 if cand_state in INDIAN_STATE_CODES else 0.0
            penalty = subs + state_bonus
            if penalty < min_penalty:
                min_penalty = penalty
                best_candidate = candidate_str

    if best_candidate is not None:
        return best_candidate, True

    return clean, False


class LicensePlateOCR:
    """
    Production-grade License Plate Optical Character Recognition (OCR) engine.

    Features:
    - PaddleOCR recognition-only inference (rec=True, det=False, cls=True)
    - Fixed CPU inference runtime bypassing OneDNN fused_conv2d defects
    - Non-aggressive preprocessing (aspect-preserved height normalization to 48px,
      8% border padding, gentle LAB-space CLAHE)
    - Fallback verification using unmodified original plate crops
    - Position-aware Indian license plate character disambiguation
    - Structured JSON-compatible output format
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        target_height: int = 48,
        padding_ratio: float = 0.08,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 1.5,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        enable_fallback: bool = True,
        use_gpu: bool = False,
        lang: str = "en",
    ) -> None:
        """
        Initializes the LicensePlateOCR engine.

        Args:
            confidence_threshold: Minimum confidence score [0.0, 1.0] to accept text. Default 0.50.
            target_height: Normalized plate height in pixels. Default 48.
            padding_ratio: Fractional padding added around the crop. Default 0.08 (8%).
            enable_clahe: Whether to apply gentle CLAHE contrast enhancement. Default True.
            clahe_clip_limit: CLAHE clip limit (lower values prevent character distortion). Default 1.5.
            clahe_grid_size: CLAHE tile grid size. Default (8, 8).
            enable_fallback: If True, evaluates unmodified crop if preprocessed crop fails validation.
            use_gpu: Whether to use GPU (False keeps OCR on CPU to protect 4GB VRAM). Default False.
            lang: Language dictionary for PaddleOCR. Default 'en'.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.target_height = int(target_height)
        self.padding_ratio = float(padding_ratio)
        self.enable_clahe = bool(enable_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid_size = clahe_grid_size
        self.enable_fallback = bool(enable_fallback)
        self.use_gpu = bool(use_gpu)
        self.lang = str(lang)

        # Lazy/Direct initialization of PaddleOCR
        self._ocr_engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Instantiates PaddleOCR with OneDNN disabled and recognition-only parameters."""
        try:
            from paddleocr import PaddleOCR
            # Initialize with explicit CPU parameters and show_log=False to reduce terminal clutter
            self._ocr_engine = PaddleOCR(
                lang=self.lang,
                use_gpu=self.use_gpu,
                enable_mkldnn=False,   # Critical: Disables OneDNN to prevent fused_conv2d crash
                use_angle_cls=True,    # Orientation classifier for angled/tilted plates
                show_log=False
            )
            logger.info("PaddleOCR recognition engine successfully initialized on CPU (MKLDNN=False).")
        except Exception as exc:
            logger.error(f"Failed to initialize PaddleOCR engine: {exc}")
            raise RuntimeError(f"Could not initialize PaddleOCR: {exc}") from exc

    def validate_crop(self, crop: Optional[np.ndarray]) -> bool:
        """Validates that the input crop is a non-empty, multi-channel image array."""
        if crop is None or not isinstance(crop, np.ndarray):
            return False
        if crop.size == 0 or crop.ndim != 3:
            return False
        h, w, c = crop.shape
        if h < 4 or w < 8 or c != 3:
            return False
        return True

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses a raw plate crop for optimal character recognition:
        1. Adds ~8% border padding (using BORDER_REPLICATE)
        2. Normalizes plate height to target_height (48 px) while preserving aspect ratio
        3. Applies gentle CLAHE contrast enhancement on the luminance channel (LAB space)
        """
        h, w = crop.shape[:2]

        # 1. 8% Border Padding
        pad_y = max(1, int(round(h * self.padding_ratio)))
        pad_x = max(1, int(round(w * self.padding_ratio)))
        padded = cv2.copyMakeBorder(
            crop, pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE
        )

        # 2. Aspect-Preserved Resizing to Target Height
        ph, pw = padded.shape[:2]
        scale = self.target_height / float(ph)
        target_width = max(16, int(round(pw * scale)))
        resized = cv2.resize(
            padded,
            (target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC
        )

        # 3. Gentle CLAHE Contrast Enhancement
        if self.enable_clahe:
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )
            l_enhanced = clahe.apply(l_channel)
            lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            return enhanced

        return resized

    def _recognize_single(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Executes recognition-only PaddleOCR inference on a single image array.

        Returns:
            Tuple[str, float]: (raw_recognized_text, confidence_score)
        """
        if self._ocr_engine is None:
            return "", 0.0

        try:
            # det=False: bypass DBNet text detector (YOLO already localized plate)
            # rec=True: SVTR_LCNet recognition
            # cls=True: 180-degree text angle orientation classifier
            results = self._ocr_engine.ocr(image, det=False, rec=True, cls=True)

            if not results or not results[0]:
                return "", 0.0

            first_pred = results[0][0]
            if isinstance(first_pred, tuple) and len(first_pred) >= 2:
                raw_text = str(first_pred[0]).strip()
                conf = float(first_pred[1])
                return raw_text, conf

            return "", 0.0
        except Exception as err:
            logger.warning(f"PaddleOCR recognition exception: {err}")
            return "", 0.0

    def predict(self, crop: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Recognizes and validates license plate text from an OpenCV BGR crop.

        Args:
            crop: OpenCV BGR image array containing the detected license plate.

        Returns:
            Dict containing:
                "text": str (Cleaned and disambiguated registration string)
                "confidence": float (OCR model confidence score [0.0, 1.0])
                "raw_text": str (Unprocessed OCR engine output)
                "valid_format": bool (True if text conforms to Indian plate regex)
        """
        # Return default empty structure for invalid crops
        if not self.validate_crop(crop):
            return {
                "text": "",
                "confidence": 0.0,
                "raw_text": "",
                "valid_format": False,
            }

        # Step 1: Preprocessed Crop Recognition
        preprocessed = self.preprocess(crop)
        prep_raw, prep_conf = self._recognize_single(preprocessed)
        prep_clean = clean_plate_text(prep_raw)
        prep_text, prep_valid = disambiguate_indian_plate(prep_clean)

        # Candidate selected by default
        best_text = prep_text
        best_conf = prep_conf
        best_raw = prep_raw
        best_valid = prep_valid

        # Step 2: Fallback Evaluation using Raw Unmodified Crop
        # If preprocessed crop yields invalid format, low confidence, or empty text,
        # evaluate the unmodified original crop to prevent over-filtering/artifacts.
        if self.enable_fallback:
            needs_fallback = (
                not prep_valid
                or prep_conf < self.confidence_threshold
                or len(prep_clean) == 0
            )

            if needs_fallback:
                raw_raw, raw_conf = self._recognize_single(crop)
                raw_clean = clean_plate_text(raw_raw)
                raw_text, raw_valid = disambiguate_indian_plate(raw_clean)

                # Selection logic:
                # 1. Prefer candidate with valid Indian format
                # 2. If both valid or both invalid, prefer higher confidence
                if raw_valid and not prep_valid:
                    best_text, best_conf, best_raw, best_valid = raw_text, raw_conf, raw_raw, True
                elif raw_valid and prep_valid:
                    if raw_conf > prep_conf:
                        best_text, best_conf, best_raw, best_valid = raw_text, raw_conf, raw_raw, True
                elif not raw_valid and not prep_valid:
                    if raw_conf > prep_conf:
                        best_text, best_conf, best_raw, best_valid = raw_text, raw_conf, raw_raw, False

        # If best confidence is below threshold, still report text but mark valid_format
        # accordingly if format check failed or confidence is negligible
        return {
            "text": best_text,
            "confidence": round(float(best_conf), 4),
            "raw_text": best_raw,
            "valid_format": bool(best_valid and best_conf >= self.confidence_threshold),
        }

    def __call__(self, crop: Optional[np.ndarray]) -> Dict[str, Any]:
        """Convenience functor mapping directly to predict(crop)."""
        return self.predict(crop)


def main() -> None:
    """Command-line interface for testing LicensePlateOCR on an image file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NETRA License Plate OCR CLI Tester"
    )
    parser.add_argument("image_path", type=str, help="Path to license plate crop image file")
    parser.add_argument(
        "--threshold", type=float, default=0.50, help="Confidence threshold (default: 0.50)"
    )
    parser.add_argument(
        "--no-clahe", action="store_true", help="Disable CLAHE preprocessing"
    )
    args = parser.parse_args()

    image_path = os.path.abspath(args.image_path)
    print("=" * 60)
    print("NETRA License Plate OCR — Command-Line Verification")
    print("=" * 60)
    print(f"Input Image: {image_path}")

    if not os.path.isfile(image_path):
        print(f"ERROR: Image file does not exist: {image_path}", file=sys.stderr)
        sys.exit(1)

    crop = cv2.imread(image_path)
    if crop is None:
        print(f"ERROR: Failed to read image using OpenCV: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Image Resolution: {crop.shape[1]}x{crop.shape[0]} (Channels: {crop.shape[2]})")

    ocr_engine = LicensePlateOCR(
        confidence_threshold=args.threshold,
        enable_clahe=not args.no_clahe,
    )

    result = ocr_engine.predict(crop)

    print("-" * 60)
    print(f"Recognized Text (Raw):  {result['raw_text']}")
    print(f"Cleaned Text:           {result['text']}")
    print(f"Confidence Score:       {result['confidence']:.4f}")
    print(f"Format Validity:        {result['valid_format']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
