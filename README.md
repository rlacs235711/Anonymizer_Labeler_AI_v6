# Batch Pipeline: Anonymize + MedGemma Label

Batch wrapper for:

```text
DICOM input -> anonymized DICOM output -> MedGemma body-part + contrast labels -> CSV
```

This is a research prototype and is not for clinical use.

## Folder Layout

```text
pipeline_anonymize_label_v6/
  label_creator/         # bundled DICOM rendering + MedGemma labeling helper
  input/                 # local batch inputs; contents not committed
  output/                # generated run output; not committed
  output (1)/            # created automatically if output/ is populated
```

Input can be:

- DICOM folders
- ZIPs containing DICOMs or DICOM folders
- individual DICOM files

The `input/` folder is intentionally committed empty through `input/.gitkeep`.
Raw DICOMs, generated outputs, model caches, and Python caches are ignored.

## Install

From the project root:

```bash
pip install -r pipeline_anonymize_label_v6/requirements-pipeline.txt
```

The `label_creator/` helper package is bundled in this repository. It contains
the DICOM loading/rendering utilities and MedGemma body-part labeling code used
by the batch pipeline.

## Run

Place DICOM inputs under:

```text
pipeline_anonymize_label_v6/input/
```

Run:

```bash
python -m pipeline_anonymize_label_v6.batch_pipeline
```

If your shell does not expose `python`, use the project virtual environment:

```bash
.venv/bin/python -m pipeline_anonymize_label_v6.batch_pipeline
```

By default, this reads from `pipeline_anonymize_label_v6/input/`, writes to
`pipeline_anonymize_label_v6/output/`, uses `basic` anonymization mode, and
uses pixel-mode contrast classification with 36 contrast montage tiles.

You can also pass explicit paths:

```bash
python -m pipeline_anonymize_label_v6.batch_pipeline \
  --input path/to/dicom_or_zip \
  --output pipeline_anonymize_label_v6/output \
  --anonymizer-mode basic
```

The default demo contrast settings are `--contrast-mode pixel` and
`--contrast-tiles 36`. You can still override them explicitly:

```bash
python -m pipeline_anonymize_label_v6.batch_pipeline \
  --input pipeline_anonymize_label_v6/input \
  --output pipeline_anonymize_label_v6/output \
  --anonymizer-mode basic \
  --contrast-mode pixel \
  --contrast-tiles 36
```

`--contrast-mode metadata` uses DICOM metadata first and only calls MedGemma
for contrast when metadata is unclear. `--contrast-mode pixel` forces a
MedGemma contrast decision using both the image quilt and metadata context.
`--contrast-tiles` can be `36` or `64`.
Final contrast outputs are forced into `contrast`, `noncontrast`, or
`not_applicable`; ambiguous visual cases are assigned the closest available
status with low confidence.

## Editing LLM Prompts

The MedGemma prompts live in:

```text
pipeline_anonymize_label_v6/label_creator/medgemma_inference.py
```

Edit these string constants to change model behavior:

```text
PROMPT_TEMPLATE                       # body-part labeling prompt
UNKNOWN_RETRY_TEMPLATE                # body-part retry prompt if the first answer is unknown
CONTRAST_PROMPT_TEMPLATE              # contrast prompt without explicit metadata fields
CONTRAST_WITH_METADATA_PROMPT_TEMPLATE # contrast prompt with metadata context
```

After changing a prompt, rerun the pipeline on a small test input first:

```bash
python -m pipeline_anonymize_label_v6.batch_pipeline \
  --input path/to/small_test_input \
  --output pipeline_anonymize_label_v6/prompt_test_output
```

Keep the required JSON output format in the prompt unless you also update the
parsing logic in `medgemma_inference.py`. For contrast, valid final labels are
`contrast`, `noncontrast`, and `not_applicable`.

## Outputs

Each run writes:

```text
pipeline_anonymize_label_v6/output/anonymized_dicom/
pipeline_anonymize_label_v6/output/csv/anonymizer_stage.csv
pipeline_anonymize_label_v6/output/csv/medgemma_helper_labels.csv
pipeline_anonymize_label_v6/output/csv/combined_results.csv
pipeline_anonymize_label_v6/output/csv/medgemma_helper_labels.json
pipeline_anonymize_label_v6/output/workflow_summary.json
```

If `output/` already exists and contains files, the pipeline writes to the next
available folder, such as `output (1)`, `output (2)`, and so on.

`combined_results.csv` includes anonymizer metadata followed by the generated
modality, body-part, and contrast label columns:

```text
source_series_uid
anonymized_series_uid
anonymized_study_uid
anonymizer_mode
anonymized_series_path
series_uid
input_subdirectory
modality
body_part_labels
contrast_status
```

`workflow_summary.json` records run-level timings for anonymizer output
generation, body-part labeling, contrast identification, and output writing.
It is intended to be easy to edit or extend with additional summary statistics
later.

## Anonymization Modes

### `basic`

Runs a basic local metadata de-identification fallback implemented in this
pipeline. It blanks common PHI-bearing fields and remaps UIDs.

This is useful for development, but it is not a full replacement for a validated
DICOM anonymizer.

### `preanonymized`

Copies already-anonymized DICOM input into the output folder and runs labeling.

Use this when RSNA Anonymizer or another validated tool has already processed
the DICOMs.

### `rsna`

Calls RSNA Anonymizer in headless mode with an existing project config:

```bash
python -m pipeline_anonymize_label_v6.batch_pipeline \
  --input pipeline_anonymize_label_v6/input \
  --output pipeline_anonymize_label_v6/output \
  --anonymizer-mode rsna \
  --rsna-config path/to/ProjectModel.json
```

The RSNA project config controls where anonymized files are written. For this
mode, configure RSNA Anonymizer to write into the run's `anonymized_dicom/`
folder, or use `preanonymized` mode afterward to label an already-anonymized
folder.

## Privacy

Do not commit raw DICOMs, anonymized DICOMs, model caches, or generated outputs.
For real sensitive data, prefer RSNA Anonymizer or another validated
de-identification tool before running MedGemma labeling.
