"""Configuration constants for reusable DICOM body-part labeling."""

from __future__ import annotations

from pathlib import Path

import torch


APP_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = APP_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

MEDGEMMA_MODEL_NAME = "google/medgemma-1.5-4b-it"
FALLBACK_MEDGEMMA_MODEL_NAME = "google/medgemma-4b-it"

LOCAL_MEDGEMMA_4B_IT_PATH = (
    APP_ROOT / "models_cache" / "medgemma_4b_it"
    if (APP_ROOT / "models_cache" / "medgemma_4b_it").exists()
    else APP_ROOT / "model_cache" / "medgemma_4b_it"
)
FALLBACK_LOCAL_MEDGEMMA_4B_IT_PATH = (
    PROJECT_ROOT / "prototype_v4_foundation_models" / "models_cache" / "medgemma_4b_it"
    if (PROJECT_ROOT / "prototype_v4_foundation_models" / "models_cache" / "medgemma_4b_it").exists()
    else PROJECT_ROOT / "prototype_v4_foundation_models" / "model_cache" / "medgemma_4b_it"
)

DEFAULT_NUM_SLICES = 25
DEFAULT_IMAGE_SIZE = 224
FULL_SERIES_MONTAGE_TILE_SIZE = 96
DEFAULT_CT_WINDOW_LEVEL = 40
DEFAULT_CT_WINDOW_WIDTH = 400

SUPPORTED_BODY_LABELS = [
    "brain",
    "head_neck",
    "chest_lung",
    "breast",
    "abdomen",
    "liver",
    "pancreas",
    "kidney",
    "pelvis",
    "prostate",
    "spine",
    "upper_extremity",
    "lower_extremity",
    "musculoskeletal",
    "whole_body",
    "bone",
    "cardiac",
    "thyroid",
    "vascular",
    "unknown",
]

SAFE_METADATA_FIELDS = [
    "Modality",
    "BodyPartExamined",
    "SeriesDescription",
    "StudyDescription",
    "ProtocolName",
    "ImageType",
    "ContrastBolusAgent",
    "ContrastBolusRoute",
    "ContrastBolusVolume",
    "ContrastBolusStartTime",
    "ContrastBolusIngredient",
    "AcquisitionContrast",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Rows",
    "Columns",
]

PHI_FIELDS_TO_EXCLUDE = {
    "PatientName",
    "PatientBirthDate",
    "AccessionNumber",
    "InstitutionName",
    "ReferringPhysicianName",
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
