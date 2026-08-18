"""MedGemma labeling stage for anonymized DICOM series."""

from __future__ import annotations

from pathlib import Path

try:
    from .label_creator import label_input_path
except ImportError:
    from label_creator import label_input_path


def _label_result_to_row(result: dict) -> dict:
    return {
        "series_uid": result.get("series_uid", ""),
        "modality": result.get("modality", ""),
        "body_part_labels": "|".join(result.get("body_part_labels", [])),
        "contrast_status": result.get("contrast_status", ""),
    }


def label_anonymized_output(
    anonymized_root: Path,
    contrast_mode: str = "metadata",
    contrast_tiles: int = 64,
) -> tuple[list[dict], list[dict]]:
    results = label_input_path(
        anonymized_root,
        contrast_mode=contrast_mode,
        contrast_tiles=contrast_tiles,
    )
    return results, [_label_result_to_row(result) for result in results]


def merge_anonymizer_and_label_rows(anonymizer_rows: list[dict], label_rows: list[dict]) -> list[dict]:
    anonymizer_by_series = {
        row.get("anonymized_series_uid") or row.get("source_series_uid"): row
        for row in anonymizer_rows
    }
    combined = []
    for label_row in label_rows:
        series_uid = label_row.get("series_uid", "")
        merged = dict(anonymizer_by_series.get(series_uid, {}))
        merged.update(label_row)
        combined.append(merged)
    return combined
