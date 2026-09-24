"""
NETRA — License Plate Optical Character Recognition (OCR) Package
Module: ocr
Project: SIH 2026 NETRA

Exposes:
- LicensePlateOCR: Production-grade recognition engine using PaddleOCR on CPU
- clean_plate_text: Normalization and alphanumeric cleaning of plate strings
- validate_indian_plate_format: Baseline regex validator for Indian plates
- disambiguate_indian_plate: Position-aware OCR character ambiguity resolver
- BASELINE_INDIAN_PLATE_REGEX: Regular expression definition for Indian plates
"""

from ocr.license_plate_ocr import (
    LicensePlateOCR,
    clean_plate_text,
    validate_indian_plate_format,
    disambiguate_indian_plate,
    BASELINE_INDIAN_PLATE_REGEX,
)

__all__ = [
    "LicensePlateOCR",
    "clean_plate_text",
    "validate_indian_plate_format",
    "disambiguate_indian_plate",
    "BASELINE_INDIAN_PLATE_REGEX",
]
