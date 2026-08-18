"""Safe metadata extraction and body-label rules."""

from __future__ import annotations

import re
from typing import Any

from .config import SAFE_METADATA_FIELDS, SUPPORTED_BODY_LABELS


MODALITY_VALUES = {"CT", "MR", "PT", "NM", "US", "MG", "CR", "DX", "XA", "RF", "OT", "SC"}


def normalize_text(value: Any) -> str:
    return str(value or "").upper()


def get_safe_series_metadata(series_data: dict) -> dict:
    metadata = series_data.get("metadata", {})
    return {field: metadata.get(field, "") for field in SAFE_METADATA_FIELDS if field in metadata}


def extract_modality(metadata: dict) -> str:
    modality = normalize_text(metadata.get("Modality", "")).strip()
    return modality if modality in MODALITY_VALUES else "UNKNOWN"


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def normalize_body_label(text: str) -> list[str]:
    text = normalize_text(text)
    labels: set[str] = set()
    if not text:
        return []
    if _has_any(text, ["WHOLEBODY", "WHOLE BODY", "SKULL TO THIGH", "VERTEX TO THIGH"]):
        labels.update(["whole_body", "chest_lung", "abdomen", "pelvis"])
    if _has_any(text, ["ABDOMEN PELVIS", "ABD PEL", "ABD/PEL", "ABD PELVIS"]):
        labels.update(["abdomen", "pelvis"])
    if re.search(r"\bCAP\b", text):
        labels.update(["chest_lung", "abdomen", "pelvis"])
    if _has_any(text, ["CHEST", "THORAX", "LUNG"]):
        labels.add("chest_lung")
    if _has_any(text, ["ABDOMEN", "ABDOMINAL", "ABD"]):
        labels.add("abdomen")
    if "PELVIS" in text or "PELVIC" in text:
        labels.add("pelvis")
    if "BRAIN" in text:
        labels.add("brain")
    if "HEAD NECK" in text or "HEADNECK" in text or "NECK" in text:
        labels.add("head_neck")
        if "HEAD" in text or "BRAIN" in text:
            labels.add("brain")
    elif "HEAD" in text:
        labels.add("brain")
    if _has_any(text, ["BREAST", "MAMMO"]):
        labels.add("breast")
    if "PROSTATE" in text:
        labels.update(["pelvis", "prostate"])
    if _has_any(text, ["SPINE", "CERVICAL", "THORACIC", "LUMBAR"]):
        labels.add("spine")
    if _has_any(text, ["KNEE", "ANKLE", "FOOT", "HIP", "LEG"]):
        labels.update(["lower_extremity", "musculoskeletal"])
    if _has_any(text, ["SHOULDER", "ARM", "ELBOW", "WRIST", "HAND"]):
        labels.update(["upper_extremity", "musculoskeletal"])
    if _has_any(text, ["LIVER", "HEPATIC"]):
        labels.add("liver")
    if "PANCREAS" in text or "PANCREATIC" in text:
        labels.add("pancreas")
    if _has_any(text, ["KIDNEY", "RENAL"]):
        labels.add("kidney")
    if _has_any(text, ["BONE", "SKELETAL"]):
        labels.add("bone")
    if _has_any(text, ["CARDIAC", "HEART", "CORONARY"]):
        labels.add("cardiac")
    if "THYROID" in text:
        labels.add("thyroid")
    if _has_any(text, ["ANGIO", "VASCULAR", "ARTERY", "VENOUS", "AORTA", "CTA"]):
        labels.add("vascular")
    return sorted(label for label in labels if label in SUPPORTED_BODY_LABELS)


def extract_metadata_body_labels(metadata: dict) -> list[str]:
    fields = ["BodyPartExamined", "SeriesDescription", "StudyDescription", "ProtocolName"]
    labels: set[str] = set()
    for field in fields:
        labels.update(normalize_body_label(metadata.get(field, "")))
    return sorted(labels) if labels else ["unknown"]


def compare_metadata_to_ai(metadata_labels: list[str], ai_labels: list[str]) -> dict:
    metadata_set = set(metadata_labels) - {"unknown"}
    ai_set = set(ai_labels) - {"unknown"}
    if not metadata_set:
        return {"conflict": False, "summary": "Metadata body label unavailable."}
    if not ai_set:
        return {"conflict": False, "summary": "AI body label unavailable."}
    overlap = metadata_set & ai_set
    return {
        "conflict": not bool(overlap),
        "summary": "Metadata and AI labels overlap." if overlap else "Metadata and AI labels differ.",
    }
