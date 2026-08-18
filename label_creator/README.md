# Label Creator

Reusable MedGemma-based DICOM series labeler.

This folder is intended to be called by the future anonymization/labeling
pipeline. It does not include the Streamlit viewer from prototype v4.

This is a research prototype and is not for clinical use.

## Purpose

Input:

```text
DICOM folder, DICOM ZIP, or individual DICOM file
```

Output:

```text
modality
body-part labels
metadata-derived body labels
safe series metadata
warnings
```

## Install

From the project root:

```bash
pip install -r label_creator/requirements-labeler.txt
```

## Model Files

The labeler first looks for MedGemma under:

```text
label_creator/models_cache/medgemma_4b_it/
label_creator/model_cache/medgemma_4b_it/
```

If not found, it falls back to the existing prototype v4 cache:

```text
prototype_v4_foundation_models/models_cache/medgemma_4b_it/
prototype_v4_foundation_models/model_cache/medgemma_4b_it/
```

If no local model is available, Hugging Face Transformers will try to load:

```text
google/medgemma-1.5-4b-it
```

Some MedGemma models require accepting model terms and authenticating locally:

```bash
hf auth login
```

## Command Line

Print JSON to terminal:

```bash
python -m label_creator.label_series path/to/dicom_input
```

Save CSV:

```bash
python -m label_creator.label_series path/to/dicom_input --csv outputs/labels.csv
```

Save JSON and CSV:

```bash
python -m label_creator.label_series path/to/dicom_input \
  --json outputs/labels.json \
  --csv outputs/labels.csv
```

## Python API

```python
from label_creator import label_input_path

results = label_input_path("path/to/dicom_input")
```

Each result is one DICOM series.

## Labeling Logic

The labeler:

1. Recursively scans for valid DICOM image files.
2. Groups files by `SeriesInstanceUID`.
3. Loads each series into a sorted image volume.
4. Selects adaptive representative slices:
   - broad coverage across the series
   - extra high-change slices based on mean absolute image difference
5. Renders selected slices into montage page(s).
6. Sends montage image(s) to MedGemma.
7. Deduplicates body-part labels and applies conservative consistency rules.

## Privacy

The labeler does not anonymize DICOM files. It avoids saving/displaying common
PHI-bearing metadata fields, but de-identification should happen upstream with
RSNA Anonymizer or another DICOM anonymization tool.
