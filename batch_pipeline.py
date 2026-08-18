"""Batch anonymize DICOM inputs and append MedGemma body-part labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .anonymizer_bridge import basic_anonymize_dicom_tree, copy_preanonymized_input, run_rsna_anonymizer
from .medgemma_labeler import label_anonymized_output, merge_anonymizer_and_label_rows
from .pipeline_utils import discover_series, ensure_dir, next_available_output_dir, prepare_input_tree, write_csv


INPUT_SUBDIRECTORY_COLUMN = "input_subdirectory"
LABEL_OUTPUT_COLUMNS = [
    INPUT_SUBDIRECTORY_COLUMN,
    "modality",
    "body_part_labels",
    "contrast_status",
]


def _ordered_combined_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in LABEL_OUTPUT_COLUMNS and key not in fieldnames:
                fieldnames.append(key)
    return fieldnames + [key for key in LABEL_OUTPUT_COLUMNS if any(key in row for row in rows)]


def _first_input_subdirectory(prepared_root: Path, paths: list[Path]) -> str:
    if not paths:
        return ""
    try:
        relative = paths[0].relative_to(prepared_root)
    except ValueError:
        return ""
    return relative.parts[0] if len(relative.parts) > 1 else ""


def _series_input_subdirectory_map(prepared_root: Path) -> dict[str, str]:
    return {
        series_uid: _first_input_subdirectory(prepared_root, paths)
        for series_uid, paths in discover_series(prepared_root).items()
    }


def _append_input_subdirectory_to_combined_rows(
    combined_rows: list[dict],
    input_subdirectory_by_series_uid: dict[str, str],
) -> None:
    for row in combined_rows:
        source_series_uid = row.get("source_series_uid", "")
        series_uid = row.get("series_uid", "")
        row[INPUT_SUBDIRECTORY_COLUMN] = (
            input_subdirectory_by_series_uid.get(source_series_uid)
            or input_subdirectory_by_series_uid.get(series_uid)
            or ""
        )


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    anonymizer_mode: str,
    rsna_config: Path | None = None,
    rsna_executable: str = "rsna-anonymizer",
    create_numbered_output: bool = True,
    contrast_mode: str = "pixel",
    contrast_tiles: int = 36,
) -> dict:
    workflow_start = time.perf_counter()
    run_output_dir = next_available_output_dir(output_dir) if create_numbered_output else ensure_dir(output_dir)
    anonymized_output_dir = ensure_dir(run_output_dir / "anonymized_dicom")
    outputs_dir = ensure_dir(run_output_dir / "csv")
    prepared_root, tempdir = prepare_input_tree(input_path)
    try:
        input_subdirectory_by_series_uid = _series_input_subdirectory_map(prepared_root)
        anonymizer_start = time.perf_counter()
        if anonymizer_mode == "rsna":
            if rsna_config is None:
                raise ValueError("--rsna-config is required when --anonymizer-mode rsna")
            completed = run_rsna_anonymizer(rsna_config, executable=rsna_executable)
            anonymizer_rows = [
                {
                    "anonymizer_mode": "rsna",
                    "rsna_config": str(rsna_config),
                    "rsna_stdout": completed.stdout,
                    "rsna_stderr": completed.stderr,
                }
            ]
            # RSNA project config must write anonymized files into this run's
            # anonymized_dicom folder for the labeling stage to scan them.
        elif anonymizer_mode == "basic":
            anonymizer_rows = basic_anonymize_dicom_tree(prepared_root, anonymized_output_dir)
        elif anonymizer_mode == "preanonymized":
            anonymizer_rows = copy_preanonymized_input(prepared_root, anonymized_output_dir)
        else:
            raise ValueError(f"Unsupported anonymizer mode: {anonymizer_mode}")
        anonymizer_seconds = time.perf_counter() - anonymizer_start

        label_start = time.perf_counter()
        label_results, label_rows = label_anonymized_output(
            anonymized_output_dir,
            contrast_mode=contrast_mode,
            contrast_tiles=contrast_tiles,
        )
        label_seconds = time.perf_counter() - label_start

        merge_start = time.perf_counter()
        combined_rows = merge_anonymizer_and_label_rows(anonymizer_rows, label_rows)
        _append_input_subdirectory_to_combined_rows(combined_rows, input_subdirectory_by_series_uid)
        merge_seconds = time.perf_counter() - merge_start

        anonymizer_csv = outputs_dir / "anonymizer_stage.csv"
        labels_csv = outputs_dir / "medgemma_helper_labels.csv"
        combined_csv = outputs_dir / "combined_results.csv"
        labels_json = outputs_dir / "medgemma_helper_labels.json"
        timing_json = run_output_dir / "workflow_summary.json"

        write_start = time.perf_counter()
        write_csv(anonymizer_rows, anonymizer_csv)
        write_csv(label_rows, labels_csv)
        write_csv(combined_rows, combined_csv, fieldnames=_ordered_combined_fieldnames(combined_rows))
        labels_json.write_text(json.dumps(label_results, indent=2))
        write_seconds = time.perf_counter() - write_start

        body_label_seconds = sum(float(result.get("body_label_seconds", 0.0) or 0.0) for result in label_results)
        contrast_seconds = sum(float(result.get("contrast_seconds", 0.0) or 0.0) for result in label_results)
        timing_summary = {
            "input_path": str(input_path),
            "output_dir": str(run_output_dir),
            "anonymizer_mode": anonymizer_mode,
            "contrast_mode": contrast_mode,
            "contrast_tiles": contrast_tiles,
            "n_anonymizer_rows": len(anonymizer_rows),
            "n_series_labeled": len(label_rows),
            "timings_seconds": {
                "workflow_total": time.perf_counter() - workflow_start,
                "anonymizer_outputs": anonymizer_seconds,
                "labeling_total": label_seconds,
                "body_part_labeling": body_label_seconds,
                "contrast_identification": contrast_seconds,
                "merge_outputs": merge_seconds,
                "write_outputs": write_seconds,
            },
            "notes": [],
        }
        timing_json.write_text(json.dumps(timing_summary, indent=2))

        return {
            "output_dir": str(run_output_dir),
            "anonymized_output_dir": str(anonymized_output_dir),
            "anonymizer_csv": str(anonymizer_csv),
            "labels_csv": str(labels_csv),
            "combined_csv": str(combined_csv),
            "labels_json": str(labels_json),
            "workflow_summary": str(timing_json),
            "n_series_labeled": len(label_rows),
        }
    finally:
        tempdir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anonymization + MedGemma labeling pipeline.")
    parser.add_argument("--input", default="pipeline_anonymize_label_v6/input", help="Input folder, ZIP, or DICOM file.")
    parser.add_argument("--output", default="pipeline_anonymize_label_v6/output", help="Output folder. If populated, output (1), output (2), etc. will be used.")
    parser.add_argument(
        "--anonymizer-mode",
        choices=["basic", "preanonymized", "rsna"],
        default="basic",
        help="basic: local metadata cleaner; preanonymized: copy only; rsna: call RSNA Anonymizer config.",
    )
    parser.add_argument("--rsna-config", help="Path to RSNA Anonymizer ProjectModel.json for headless mode.")
    parser.add_argument("--rsna-executable", default="rsna-anonymizer")
    parser.add_argument(
        "--contrast-mode",
        choices=["metadata", "pixel"],
        default="pixel",
        help="metadata: metadata-first with VLM fallback when unclear; pixel: always run VLM with metadata context.",
    )
    parser.add_argument(
        "--contrast-tiles",
        type=int,
        choices=[36, 64],
        default=36,
        help="Maximum tiles per contrast montage page.",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        anonymizer_mode=args.anonymizer_mode,
        rsna_config=Path(args.rsna_config) if args.rsna_config else None,
        rsna_executable=args.rsna_executable,
        contrast_mode=args.contrast_mode,
        contrast_tiles=args.contrast_tiles,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
