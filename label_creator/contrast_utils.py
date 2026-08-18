"""Contrast-status classification helpers."""

from __future__ import annotations

import re
from typing import Any

from .metadata_utils import normalize_text


CONTRAST_APPLICABLE_MODALITIES = {"CT", "MR", "PT", "NM", "US", "XA", "RF"}
CONTRAST_STATUS_VALUES = {"contrast", "noncontrast", "not_applicable"}

CONTRAST_METADATA_FIELDS = [
    "ContrastBolusAgent",
    "ContrastBolusRoute",
    "ContrastBolusVolume",
    "ContrastBolusStartTime",
    "ContrastBolusIngredient",
    "AcquisitionContrast",
    "ImageType",
    "SeriesDescription",
    "StudyDescription",
    "ProtocolName",
]

NEGATIVE_PATTERNS = [
    r"\bW/?O\b",
    r"\bWO\b",
    r"\bWITHOUT\b",
    r"\bNON[- ]?CON\b",
    r"\bNONCON\b",
    r"\bNO\s+(IV\s+)?CONTRAST\b",
    r"\bPRE[- ]?CONTRAST\b",
    r"\bPRE\b",
    r"\bPLAIN\b",
    r"\bUNENHANCED\b",
]

POSITIVE_PATTERNS = [
    r"\bW/\b",
    r"\bWITH\s+(IV\s+)?CONTRAST\b",
    r"\bWITH\b",
    r"\bPOST[- ]?CONTRAST\b",
    r"\bPOST\b",
    r"\bENHANCED\b",
    r"\bCONTRAST\b",
    r"\bIVC\b",
    r"\bCE\b",
    r"\bC\+\b",
    r"\bGAD\b",
    r"\bGADOLINIUM\b",
    r"\bDYNAMIC\b",
    r"\bARTERIAL\b",
    r"\bVENOUS\b",
    r"\bPORTAL\b",
    r"\bDELAY(?:ED)?\b",
    r"\bCTA\b",
    r"\bANGIO\b",
]

STRONG_POSITIVE_PATTERNS = [
    r"\bPOST[- ]?CONTRAST\b",
    r"\bENHANCED\b",
    r"\bIVC\b",
    r"\bC\+\b",
    r"\bGAD\b",
    r"\bGADOLINIUM\b",
    r"\bDYNAMIC\b",
    r"\bARTERIAL\b",
    r"\bVENOUS\b",
    r"\bPORTAL\b",
    r"\bDELAY(?:ED)?\b",
    r"\bCTA\b",
    r"\bANGIO\b",
]


def _metadata_text(metadata: dict[str, Any]) -> str:
    return " ".join(normalize_text(metadata.get(field, "")) for field in CONTRAST_METADATA_FIELDS)


def _has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def normalize_contrast_status(value: str) -> str:
    value = str(value or "").strip().lower()
    if value in {"contrast", "contrast_enhanced", "enhanced", "postcontrast", "post-contrast"}:
        return "contrast"
    if value in {"noncontrast", "non_contrast", "no_contrast", "without_contrast", "unenhanced"}:
        return "noncontrast"
    if value in {"not_applicable", "not applicable", "na", "n/a"}:
        return "not_applicable"
    return value if value in CONTRAST_STATUS_VALUES else "noncontrast"


def classify_contrast_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    modality = normalize_text(metadata.get("Modality", "")).strip()
    if modality and modality not in CONTRAST_APPLICABLE_MODALITIES:
        return {
            "contrast_status": "not_applicable",
            "contrast_confidence": "high",
            "contrast_evidence": "metadata",
            "contrast_metadata_basis": f"Modality {modality} is not routinely contrast-classified.",
        }

    agent = normalize_text(metadata.get("ContrastBolusAgent", "")).strip()
    ingredient = normalize_text(metadata.get("ContrastBolusIngredient", "")).strip()
    route = normalize_text(metadata.get("ContrastBolusRoute", "")).strip()
    volume = normalize_text(metadata.get("ContrastBolusVolume", "")).strip()
    acquisition_contrast = normalize_text(metadata.get("AcquisitionContrast", "")).strip()
    if any(value for value in [agent, ingredient, route, volume]):
        return {
            "contrast_status": "contrast",
            "contrast_confidence": "high",
            "contrast_evidence": "metadata",
            "contrast_metadata_basis": "Contrast bolus metadata is populated.",
        }
    if acquisition_contrast in {"CONTRAST", "CONTRASTENHANCED", "CONTRAST_ENHANCED"}:
        return {
            "contrast_status": "contrast",
            "contrast_confidence": "high",
            "contrast_evidence": "metadata",
            "contrast_metadata_basis": "AcquisitionContrast indicates contrast.",
        }

    text = _metadata_text(metadata)
    has_negative = _has_pattern(text, NEGATIVE_PATTERNS)
    has_positive = _has_pattern(text, POSITIVE_PATTERNS)
    has_strong_positive = _has_pattern(text, STRONG_POSITIVE_PATTERNS)
    if has_positive and not has_negative:
        return {
            "contrast_status": "contrast",
            "contrast_confidence": "medium",
            "contrast_evidence": "metadata",
            "contrast_metadata_basis": "Series/study/protocol text suggests contrast.",
        }
    if has_negative and not has_strong_positive:
        return {
            "contrast_status": "noncontrast",
            "contrast_confidence": "medium",
            "contrast_evidence": "metadata",
            "contrast_metadata_basis": "Series/study/protocol text suggests no contrast.",
        }
    if has_positive and has_negative:
        return {
            "contrast_status": "unknown",
            "contrast_confidence": "low",
            "contrast_evidence": "metadata",
            "contrast_metadata_basis": "Metadata contains both contrast and noncontrast terms.",
        }
    return {
        "contrast_status": "unknown",
        "contrast_confidence": "low",
        "contrast_evidence": "metadata",
        "contrast_metadata_basis": "No decisive contrast metadata found.",
    }
