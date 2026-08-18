"""Bridge for RSNA Anonymizer or basic local DICOM metadata anonymization."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pydicom
from pydicom.uid import generate_uid

from .pipeline_utils import discover_series, ensure_dir


PHI_TAGS_TO_BLANK = [
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _relative_series_path(series_uid: str) -> Path:
    safe_uid = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in series_uid)
    return Path(safe_uid or "unknown_series")


def run_rsna_anonymizer(
    config_path: Path,
    executable: str = "rsna-anonymizer",
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess:
    """Run RSNA Anonymizer headlessly using an existing project config."""
    command = [executable, "-c", str(config_path)]
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)


def basic_anonymize_dicom_tree(input_root: Path, output_root: Path) -> list[dict]:
    """Basic metadata de-identification fallback; not a replacement for RSNA Anonymizer."""
    ensure_dir(output_root)
    series_map = discover_series(input_root)
    uid_map: dict[str, str] = {}
    rows = []

    for source_series_uid, paths in sorted(series_map.items()):
        anonymized_series_uid = uid_map.setdefault(source_series_uid, generate_uid())
        series_dir = output_root / _relative_series_path(anonymized_series_uid)
        series_dir.mkdir(parents=True, exist_ok=True)
        anonymized_study_uid = ""

        for index, path in enumerate(paths, start=1):
            try:
                ds = pydicom.dcmread(path, force=True)
            except Exception:
                continue

            source_study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            anonymized_study_uid = uid_map.setdefault(source_study_uid, generate_uid()) if source_study_uid else ""
            source_patient_id = str(getattr(ds, "PatientID", "") or "")

            if source_patient_id:
                ds.PatientID = f"ANON_{_hash_text(source_patient_id)}"
            else:
                ds.PatientID = "ANON_UNKNOWN"
            for tag_name in PHI_TAGS_TO_BLANK:
                if hasattr(ds, tag_name):
                    setattr(ds, tag_name, "")
            if anonymized_study_uid:
                ds.StudyInstanceUID = anonymized_study_uid
            ds.SeriesInstanceUID = anonymized_series_uid
            if hasattr(ds, "SOPInstanceUID"):
                ds.SOPInstanceUID = generate_uid()

            destination = series_dir / f"image_{index:06d}.dcm"
            ds.save_as(destination, write_like_original=False)

        rows.append(
            {
                "source_series_uid": source_series_uid,
                "anonymized_series_uid": anonymized_series_uid,
                "anonymized_study_uid": anonymized_study_uid,
                "anonymized_series_path": str(series_dir),
                "anonymizer_mode": "basic_metadata_fallback",
            }
        )
    return rows


def copy_preanonymized_input(input_root: Path, output_root: Path) -> list[dict]:
    """Copy already-anonymized DICOMs into the output folder for labeling."""
    ensure_dir(output_root)
    if output_root.exists():
        shutil.copytree(input_root, output_root, dirs_exist_ok=True)
    rows = []
    for series_uid, paths in discover_series(output_root).items():
        rows.append(
            {
                "source_series_uid": series_uid,
                "anonymized_series_uid": series_uid,
                "anonymized_series_path": str(paths[0].parent),
                "anonymizer_mode": "already_anonymized_copy",
            }
        )
    return rows
