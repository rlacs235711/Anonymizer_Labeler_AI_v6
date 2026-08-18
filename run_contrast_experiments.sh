#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

INPUT_PATH="${INPUT_PATH:-pipeline_anonymize_label_v6/Chest_Input}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-pipeline_anonymize_label_v6/chest_output_contrast}"
ANONYMIZER_MODE="${ANONYMIZER_MODE:-basic}"

for contrast_mode in metadata pixel; do
  for contrast_tiles in 36 64; do
    output_dir="${OUTPUT_PREFIX}_${contrast_mode}_${contrast_tiles}"
    echo "Running contrast_mode=${contrast_mode}, contrast_tiles=${contrast_tiles}"
    "${PYTHON_BIN}" -m pipeline_anonymize_label_v6.batch_pipeline \
      --input "${INPUT_PATH}" \
      --output "${output_dir}" \
      --anonymizer-mode "${ANONYMIZER_MODE}" \
      --contrast-mode "${contrast_mode}" \
      --contrast-tiles "${contrast_tiles}"
  done
done
