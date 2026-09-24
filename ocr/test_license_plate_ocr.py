"""
NETRA — Unit & Functional Tests for License Plate OCR Engine
File: ocr/test_license_plate_ocr.py
Component: Model 3 (Plate OCR)
Project: SIH 2026 NETRA
"""

import os
import unittest
import numpy as np
import cv2

from ocr.license_plate_ocr import (
    LicensePlateOCR,
    clean_plate_text,
    validate_indian_plate_format,
    disambiguate_indian_plate,
)


class TestLicensePlateOCR(unittest.TestCase):
    """Unit and functional tests for the LicensePlateOCR module."""

    @classmethod
    def setUpClass(cls):
        """Initialize the OCR engine once for all tests to minimize initialization overhead."""
        cls.ocr_engine = LicensePlateOCR(
            confidence_threshold=0.50,
            target_height=48,
            padding_ratio=0.08,
            enable_clahe=True,
            enable_fallback=True,
            use_gpu=False,
            lang="en"
        )
        cls.sample_plate_path = os.path.join(
            os.path.dirname(__file__), "test_samples", "sample_plate_gj01.jpg"
        )

    def test_clean_plate_text(self):
        """Verify text normalization, whitespace removal, and punctuation stripping."""
        # 1. Lowercase to uppercase
        self.assertEqual(clean_plate_text("mh12de1433"), "MH12DE1433")
        # 2. Punctuation removal (hyphens, dots, spaces)
        self.assertEqual(clean_plate_text("TN-04 AZ 6643"), "TN04AZ6643")
        self.assertEqual(clean_plate_text("GJ.01-WC_8529"), "GJ01WC8529")
        # 3. Non-alphanumeric symbols
        self.assertEqual(clean_plate_text("[DL-1C-AB:1234]"), "DL1CAB1234")
        # 4. None and empty string
        self.assertEqual(clean_plate_text(""), "")
        self.assertEqual(clean_plate_text(None), "")

    def test_validate_indian_plate_format(self):
        """Verify baseline Indian license plate regular expression validation."""
        # Valid standard civilian/commercial formats
        self.assertTrue(validate_indian_plate_format("MH12DE1433"))
        self.assertTrue(validate_indian_plate_format("DL1CAB1234"))
        self.assertTrue(validate_indian_plate_format("GJ01WC8529"))
        self.assertTrue(validate_indian_plate_format("KA531234"))
        self.assertTrue(validate_indian_plate_format("TN04AZ6643"))
        self.assertTrue(validate_indian_plate_format("UP32MN0001"))

        # Invalid formats
        self.assertFalse(validate_indian_plate_format("ABC"))              # Too short
        self.assertFalse(validate_indian_plate_format("MH123456789012"))  # Too long
        self.assertFalse(validate_indian_plate_format("12MH1234"))        # Starts with numbers
        self.assertFalse(validate_indian_plate_format("MH12DE123"))       # Only 3 registration digits
        self.assertFalse(validate_indian_plate_format("MH12DE12345"))     # 5 registration digits
        self.assertFalse(validate_indian_plate_format(""))                # Empty

    def test_disambiguate_indian_plate(self):
        """Verify position-aware OCR ambiguity substitution (O/0, I/1, Z/2, S/5, B/8)."""
        # Case 1: Letter 'O' in numeric suffix -> replaced by '0'
        text, is_valid = disambiguate_indian_plate("MH12DE143O")
        self.assertEqual(text, "MH12DE1430")
        self.assertTrue(is_valid)

        # Case 2: Digit '8' in series letters -> replaced by 'B'
        text, is_valid = disambiguate_indian_plate("DL1CA81234")
        self.assertEqual(text, "DL1CAB1234")
        self.assertTrue(is_valid)

        # Case 3: Letter 'I' in numeric suffix -> replaced by '1'
        text, is_valid = disambiguate_indian_plate("KA29Z999I")
        self.assertEqual(text, "KA29Z9991")
        self.assertTrue(is_valid)

        # Case 4: Digit '8' in series letters -> replaced by 'B'
        text, is_valid = disambiguate_indian_plate("MH028T6482")
        self.assertEqual(text, "MH02BT6482")
        self.assertTrue(is_valid)

        # Case 5: Letter 'S' in numeric suffix -> replaced by '5'
        text, is_valid = disambiguate_indian_plate("GJ01WC852S")
        self.assertEqual(text, "GJ01WC8525")
        self.assertTrue(is_valid)

        # Case 6: Non-plate string cannot be disambiguated to valid
        text, is_valid = disambiguate_indian_plate("XYZNONVALID")
        self.assertFalse(is_valid)

    def test_empty_and_invalid_crop_handling(self):
        """Verify graceful error handling for None, empty, and invalid crop dimensions."""
        # Test None
        res_none = self.ocr_engine.predict(None)
        self.assertFalse(res_none["valid_format"])
        self.assertEqual(res_none["confidence"], 0.0)
        self.assertEqual(res_none["text"], "")
        self.assertEqual(res_none["raw_text"], "")

        # Test empty numpy array
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        res_empty = self.ocr_engine.predict(empty_crop)
        self.assertFalse(res_empty["valid_format"])
        self.assertEqual(res_empty["confidence"], 0.0)
        self.assertEqual(res_empty["text"], "")

        # Test microscopic/invalid dimensions (e.g. 2x2)
        tiny_crop = np.zeros((2, 2, 3), dtype=np.uint8)
        res_tiny = self.ocr_engine.predict(tiny_crop)
        self.assertFalse(res_tiny["valid_format"])
        self.assertEqual(res_tiny["confidence"], 0.0)

    def test_ocr_result_structure(self):
        """Verify the exact dictionary structure and data types of OCR predictions."""
        dummy_crop = np.full((48, 160, 3), 200, dtype=np.uint8)
        result = self.ocr_engine.predict(dummy_crop)

        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertIn("raw_text", result)
        self.assertIn("valid_format", result)

        self.assertIsInstance(result["text"], str)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["raw_text"], str)
        self.assertIsInstance(result["valid_format"], bool)

    def test_preprocessing_pipeline(self):
        """Verify aspect-preserved resizing and output shape from preprocessing."""
        crop = np.zeros((56, 189, 3), dtype=np.uint8)
        preprocessed = self.ocr_engine.preprocess(crop)

        # Height must be normalized to target_height (48)
        self.assertEqual(preprocessed.shape[0], 48)
        # Channels must remain 3
        self.assertEqual(preprocessed.shape[2], 3)
        # Width must be proportional (> 100 px)
        self.assertGreater(preprocessed.shape[1], 100)

    def test_real_plate_crop_inference(self):
        """Verify end-to-end OCR recognition on real Indian license plate crop."""
        self.assertTrue(
            os.path.isfile(self.sample_plate_path),
            f"Sample plate file not found at: {self.sample_plate_path}"
        )

        crop = cv2.imread(self.sample_plate_path)
        self.assertIsNotNone(crop, "Failed to load sample plate image.")

        result = self.ocr_engine.predict(crop)

        # Assert recognition quality on known ground truth (GJ01WC8529)
        self.assertEqual(result["text"], "GJ01WC8529")
        self.assertGreaterEqual(result["confidence"], 0.50)
        self.assertTrue(result["valid_format"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
