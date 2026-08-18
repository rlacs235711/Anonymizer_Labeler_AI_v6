"""DICOM loading, ZIP handling, and series grouping utilities."""

from __future__ import annotations

import hashlib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from .config import PHI_FIELDS_TO_EXCLUDE, SAFE_METADATA_FIELDS


UNSUPPORTED_MODALITIES = {"SEG", "RTSTRUCT", "RTDOSE", "RTPLAN", "SR", "PR", "KO", "SM"}
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _vector(value: Any, expected_len: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray([float(v) for v in value], dtype=np.float32)
    except (TypeError, ValueError):
        return None
    return arr if arr.size == expected_len else None


def is_dicom_file(path: Path) -> bool:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return False
    modality = str(getattr(ds, "Modality", "") or "").upper()
    return bool(getattr(ds, "SeriesInstanceUID", None)) and modality not in UNSUPPORTED_MODALITIES


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def read_metadata(path: Path) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    metadata = {field: str(getattr(ds, field, "") or "") for field in SAFE_METADATA_FIELDS}
    metadata["Rows"] = _as_int(getattr(ds, "Rows", None))
    metadata["Columns"] = _as_int(getattr(ds, "Columns", None))
    patient_id = str(getattr(ds, "PatientID", "") or "")
    if patient_id:
        metadata["HashedPatientID"] = _hash_value(patient_id)
    for field in PHI_FIELDS_TO_EXCLUDE:
        metadata.pop(field, None)
    return metadata


def safe_extract_zip(uploaded_zip, output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    total_size = 0
    with zipfile.ZipFile(uploaded_zip) as zf:
        for info in zf.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP path rejected: {info.filename}")
            total_size += int(info.file_size)
            if total_size > MAX_EXTRACTED_BYTES:
                raise ValueError("ZIP extraction would exceed the 2 GB safety limit.")
            destination = (output_dir / member).resolve()
            if not str(destination).startswith(str(output_dir)):
                raise ValueError(f"Unsafe ZIP destination rejected: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, destination.open("wb") as target:
                target.write(source.read())
    return output_dir


def collect_dicom_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and is_dicom_file(path)]


def group_by_series(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        try:
            uid = read_metadata(path).get("SeriesInstanceUID")
        except Exception:
            continue
        if uid:
            grouped[uid].append(path)
    return dict(grouped)


def choose_best_series(series_map: dict[str, list[Path]]) -> str:
    if not series_map:
        raise ValueError("No valid DICOM image series found.")
    return max(series_map, key=lambda uid: len(series_map[uid]))


def _sort_key(ds: pydicom.Dataset, path: Path) -> tuple[int, float | int | str, str]:
    position = _vector(getattr(ds, "ImagePositionPatient", None), 3)
    orientation = _vector(getattr(ds, "ImageOrientationPatient", None), 6)
    if position is not None and orientation is not None:
        normal = np.cross(orientation[:3], orientation[3:])
        return (0, float(np.dot(position, normal)), str(path))
    slice_location = _as_float(getattr(ds, "SliceLocation", None))
    if slice_location is not None:
        return (1, slice_location, str(path))
    instance_number = _as_int(getattr(ds, "InstanceNumber", None))
    if instance_number is not None:
        return (2, instance_number, str(path))
    return (3, str(path), str(path))


def sort_datasets(datasets: list[tuple[pydicom.Dataset, Path]]) -> list[tuple[pydicom.Dataset, Path]]:
    return sorted(datasets, key=lambda item: _sort_key(item[0], item[1]))


def apply_rescale(ds, arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope != 1 or intercept != 0:
        arr = arr * slope + intercept
    return arr


def _frames_from_dataset(ds: pydicom.Dataset, path: Path, warnings: list[str]) -> list[np.ndarray]:
    try:
        pixels = np.asarray(ds.pixel_array)
    except Exception as exc:
        warnings.append(f"Pixel decode failed for {path.name}: {exc}")
        return []
    if pixels.ndim == 2:
        frames = [pixels]
    elif pixels.ndim == 3:
        frames = [pixels[i] for i in range(pixels.shape[0])]
    elif pixels.ndim == 4:
        frames = [pixels[i, ..., 0] for i in range(pixels.shape[0])]
    else:
        warnings.append(f"Unsupported pixel shape {pixels.shape} in {path.name}")
        return []
    return [apply_rescale(ds, frame) for frame in frames]


def load_series(series_paths: list[Path]) -> dict:
    datasets = []
    warnings: list[str] = []
    metadata = {}
    monochrome1 = False
    for path in series_paths:
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception as exc:
            warnings.append(f"Read failed for {path.name}: {exc}")
            continue
        modality = str(getattr(ds, "Modality", "") or "").upper()
        if modality in UNSUPPORTED_MODALITIES or "PixelData" not in ds:
            continue
        datasets.append((ds, path))
        monochrome1 = monochrome1 or str(getattr(ds, "PhotometricInterpretation", "") or "").upper() == "MONOCHROME1"
        if not metadata:
            metadata = read_metadata(path)

    frames = []
    for ds, path in sort_datasets(datasets):
        frames.extend(_frames_from_dataset(ds, path, warnings))
    if not frames:
        raise ValueError("No readable image frames found in selected series.")

    common_shape, _ = Counter(frame.shape for frame in frames).most_common(1)[0]
    kept = [frame for frame in frames if frame.shape == common_shape]
    if len(kept) != len(frames):
        warnings.append(f"Kept {len(kept)} frames with common shape {common_shape}; skipped {len(frames) - len(kept)} mixed-shape frames.")
    volume = np.stack(kept)
    if monochrome1:
        volume = np.max(volume) - volume
    return {
        "volume": volume,
        "metadata": metadata,
        "warnings": warnings,
        "series_uid": metadata.get("SeriesInstanceUID", ""),
        "n_files": len(series_paths),
        "monochrome1": monochrome1,
    }
