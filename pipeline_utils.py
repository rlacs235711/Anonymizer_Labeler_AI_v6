"""Shared filesystem and CSV helpers for the anonymize-label pipeline."""

from __future__ import annotations

import csv
import shutil
import tempfile
import zipfile
from pathlib import Path

try:
    from .label_creator.dicom_utils import collect_dicom_files, group_by_series, is_dicom_file
except ImportError:
    from label_creator.dicom_utils import collect_dicom_files, group_by_series, is_dicom_file


MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_populated(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def next_available_output_dir(base_path: Path) -> Path:
    if not is_populated(base_path):
        return ensure_dir(base_path)
    parent = base_path.parent
    stem = base_path.name
    index = 1
    while True:
        candidate = parent / f"{stem} ({index})"
        if not is_populated(candidate):
            return ensure_dir(candidate)
        index += 1


def safe_extract_zip(zip_path: Path, output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    total_size = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP path rejected: {info.filename}")
            total_size += int(info.file_size)
            if total_size > MAX_EXTRACTED_BYTES:
                raise ValueError(f"ZIP extraction would exceed {MAX_EXTRACTED_BYTES // (1024**3)} GB.")
            destination = (output_dir / member).resolve()
            if not str(destination).startswith(str(output_dir)):
                raise ValueError(f"Unsafe ZIP destination rejected: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    return output_dir


def prepare_input_tree(input_root: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return a scan-ready folder containing folders, loose DICOMs, and extracted ZIPs."""
    tempdir = tempfile.TemporaryDirectory()
    prepared_root = Path(tempdir.name) / "prepared_input"
    prepared_root.mkdir(parents=True, exist_ok=True)

    if input_root.is_file():
        if input_root.suffix.lower() == ".zip":
            safe_extract_zip(input_root, prepared_root / input_root.stem)
        elif is_dicom_file(input_root):
            shutil.copy2(input_root, prepared_root / input_root.name)
        else:
            raise ValueError(f"Unsupported input file: {input_root}")
        return prepared_root, tempdir

    if not input_root.exists():
        raise FileNotFoundError(input_root)

    for path in input_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(input_root)
        if path.suffix.lower() == ".zip":
            safe_extract_zip(path, prepared_root / relative.with_suffix(""))
        elif is_dicom_file(path):
            destination = prepared_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return prepared_root, tempdir


def discover_series(input_root: Path) -> dict[str, list[Path]]:
    paths = collect_dicom_files(input_root)
    return group_by_series(paths)


def zip_directory(source_dir: Path, output_zip: Path) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))
    return output_zip


def write_csv(rows: list[dict], output_path: Path, fieldnames: list[str] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("")
        return
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
