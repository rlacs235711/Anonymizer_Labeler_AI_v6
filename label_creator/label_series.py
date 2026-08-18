"""Command-line entry point for labeling DICOM series with MedGemma."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .labeler import label_input_path


def _csv_row(result: dict) -> dict:
    return {
        "source_path": result.get("source_path", ""),
        "study_uid": result.get("study_uid", ""),
        "series_uid": result.get("series_uid", ""),
        "modality": result.get("modality", ""),
        "body_part_labels": "|".join(result.get("body_part_labels", [])),
        "metadata_body_labels": "|".join(result.get("metadata_body_labels", [])),
        "n_files": result.get("n_files", ""),
        "n_frames": result.get("n_frames", ""),
        "adaptive_slice_count": result.get("adaptive_slice_count", ""),
        "montage_page_count": result.get("montage_page_count", ""),
        "warnings": " | ".join(result.get("warnings", [])),
    }


def write_csv(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "study_uid",
        "series_uid",
        "modality",
        "body_part_labels",
        "metadata_body_labels",
        "n_files",
        "n_frames",
        "adaptive_slice_count",
        "montage_page_count",
        "warnings",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(_csv_row(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Label DICOM series using MedGemma.")
    parser.add_argument("input_path", help="DICOM folder, DICOM ZIP, or individual DICOM file.")
    parser.add_argument("--json", dest="json_path", help="Optional JSON output path.")
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path.")
    args = parser.parse_args()

    results = label_input_path(args.input_path)
    if args.json_path:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
    if args.csv_path:
        write_csv(results, Path(args.csv_path))
    if not args.json_path and not args.csv_path:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
