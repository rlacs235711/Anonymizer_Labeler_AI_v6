"""MedGemma prompt-based visible anatomy inference."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from .config import (
    FALLBACK_LOCAL_MEDGEMMA_4B_IT_PATH,
    FALLBACK_MEDGEMMA_MODEL_NAME,
    LOCAL_MEDGEMMA_4B_IT_PATH,
    MEDGEMMA_MODEL_NAME,
    SUPPORTED_BODY_LABELS,
    get_device,
)
from .contrast_utils import normalize_contrast_status
from .metadata_utils import extract_metadata_body_labels, normalize_body_label


PROMPT_TEMPLATE = """You are given one rendered montage from a complete DICOM series.
The montage contains slices sampled across the full series, so review every tile before answering.
The DICOM metadata-reported modality is {modality}.
Metadata suggests possible body regions: {metadata_hint}.
Use metadata only as weak context; rely on visible anatomy in the image.

Your only task is to identify visible anatomy/body parts present anywhere in the series.
Do not diagnose disease. Do not describe findings. Do not explain your answer.

Allowed labels (STRICTLY USE THESE LABELS):
brain, head_neck, chest_lung, breast, abdomen, pelvis, spine, upper_extremity, lower_extremity, whole_body, unknown

Output rules:
- Return only labels from the allowed list.
- Include every relevant visible body-part label.
- Order labels from most relevant/prominent for the series to least.
- Do not use "unknown" if any allowed anatomy is visible.
- Use "unknown" only if no allowed body region can be identified at all.
- Return JSON only.
- Do not include confidence, reason, markdown, or any extra text.

Required output format:
{{"visible_body_regions":["label"]}}
"""

UNKNOWN_RETRY_TEMPLATE = """The previous answer was unknown.
You are given the same rendered DICOM series montage.
The DICOM metadata-reported modality is {modality}.
Metadata suggests possible body regions: {metadata_hint}.

Choose the closest visible body-region labels from the allowed list unless the montage is truly non-anatomic.
Do not diagnose disease. Do not explain your answer.

Allowed labels:
brain, head_neck, chest_lung, breast, abdomen, pelvis, spine, upper_extremity, lower_extremity, whole_body, unknown

Return JSON only:
{{"visible_body_regions":["label"]}}
"""

CONTRAST_PROMPT_TEMPLATE = """You are given one rendered montage from a DICOM series.
The montage contains slices sampled across the series, so review every tile before answering.
Your only task is to determine whether the visible images appear contrast-enhanced.

Use the pixel appearance of the montage and any provided metadata context.

Allowed statuses:
contrast, noncontrast, not_applicable

Definitions:
- contrast: visible IV/oral/gadolinium/iodinated contrast enhancement is likely present.
- noncontrast: the series appears unenhanced or without visible contrast.
- not_applicable: contrast status is not meaningful for this image type.

If visual evidence is limited or uncertain, choose the closest status between contrast and noncontrast.

Return JSON only.
Do not include diagnosis, findings, markdown, or extra text.

Required output format:
{{"contrast_status":"noncontrast","confidence":"low"}}
"""

CONTRAST_WITH_METADATA_PROMPT_TEMPLATE = """You are given one rendered montage from a DICOM series.
The montage contains slices sampled across the series, so review every tile before answering.
Your only task is to determine whether this series is contrast-enhanced.

Metadata context:
Modality: {modality}
Contrast metadata/rule result: {metadata_status}
Metadata basis: {metadata_basis}

Use metadata as useful context, but check whether the visible image appearance supports it.

Allowed statuses:
contrast, noncontrast, not_applicable

If evidence is limited or conflicting, choose the closest status between contrast and noncontrast.

Return JSON only.
Do not include diagnosis, findings, markdown, or extra text.

Required output format:
{{"contrast_status":"noncontrast","confidence":"low"}}
"""


def _model_name_or_path() -> str:
    if LOCAL_MEDGEMMA_4B_IT_PATH.exists():
        return str(LOCAL_MEDGEMMA_4B_IT_PATH)
    if FALLBACK_LOCAL_MEDGEMMA_4B_IT_PATH.exists():
        return str(FALLBACK_LOCAL_MEDGEMMA_4B_IT_PATH)
    return MEDGEMMA_MODEL_NAME or FALLBACK_MEDGEMMA_MODEL_NAME


@lru_cache(maxsize=1)
def load_medgemma_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = get_device()
    model_name = _model_name_or_path()
    processor = AutoProcessor.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, processor, device, model_name


def _extract_json(text: str) -> dict | None:
    cleaned = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced or re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, flags=re.DOTALL)
    candidates = candidates or [cleaned]
    for candidate in reversed(candidates):
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"visible_body_regions": parsed}
        except json.JSONDecodeError:
            continue
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"visible_body_regions": parsed}
    except json.JSONDecodeError:
        return None


def _normalize_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        if label in SUPPORTED_BODY_LABELS:
            normalized = [label]
        else:
            normalized = normalize_body_label(label)
        for item in normalized:
            if item not in out:
                out.append(item)
    if not out:
        return ["unknown"]
    if len(out) > 1:
        out = [label for label in out if label != "unknown"]
    return out


def _normalize_confidence(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in {"high", "medium", "low"} else "low"


def _run_medgemma_with_prompt(image: Image.Image, prompt: str) -> dict:
    try:
        model, processor, device, model_name = load_medgemma_model()
    except Exception as exc:
        return {"visible_body_regions": [], "raw_text": "", "warning": f"MedGemma load failed: {exc}"}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw_text = processor.decode(output_ids[0], skip_special_tokens=True)
    except Exception as exc:
        return {"visible_body_regions": [], "raw_text": "", "warning": f"MedGemma inference failed: {exc}"}

    parsed = _extract_json(raw_text)
    if not parsed:
        return {"visible_body_regions": [], "raw_text": raw_text, "warning": "MedGemma did not return valid JSON."}
    labels = _normalize_labels(parsed.get("visible_body_regions", []))
    return {
        "visible_body_regions": labels,
        "raw_text": raw_text,
        "model": model_name,
    }


def _run_medgemma_contrast_with_prompt(image: Image.Image, prompt: str) -> dict:
    result = _run_medgemma_with_raw_prompt(image, prompt)
    if result.get("warning"):
        return result
    parsed = _extract_json(result.get("raw_text", ""))
    if not parsed:
        return {
            "contrast_status": "noncontrast",
            "contrast_confidence": "low",
            "raw_text": result.get("raw_text", ""),
            "warning": "MedGemma did not return valid contrast JSON.",
        }
    return {
        "contrast_status": normalize_contrast_status(parsed.get("contrast_status", "")),
        "contrast_confidence": _normalize_confidence(parsed.get("confidence", "")),
        "raw_text": result.get("raw_text", ""),
        "model": result.get("model", ""),
    }


def _run_medgemma_with_raw_prompt(image: Image.Image, prompt: str) -> dict:
    try:
        model, processor, device, model_name = load_medgemma_model()
    except Exception as exc:
        return {"raw_text": "", "warning": f"MedGemma load failed: {exc}"}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw_text = processor.decode(output_ids[0], skip_special_tokens=True)
    except Exception as exc:
        return {"raw_text": "", "warning": f"MedGemma inference failed: {exc}"}
    return {"raw_text": raw_text, "model": model_name}


def run_medgemma(montage_image: Image.Image, modality: str, metadata_summary: dict) -> dict:
    metadata_labels = extract_metadata_body_labels(metadata_summary)
    metadata_hint = ", ".join(label for label in metadata_labels if label != "unknown") or "none"
    prompt = PROMPT_TEMPLATE.format(modality=modality, metadata_hint=metadata_hint)
    result = _run_medgemma_with_prompt(montage_image, prompt)
    if result.get("visible_body_regions") == ["unknown"]:
        retry_prompt = UNKNOWN_RETRY_TEMPLATE.format(modality=modality, metadata_hint=metadata_hint)
        retry = _run_medgemma_with_prompt(montage_image, retry_prompt)
        retry["retried_after_unknown"] = True
        return retry
    result["retried_after_unknown"] = False
    return result


def run_medgemma_contrast(
    montage_image: Image.Image,
    modality: str,
    metadata_result: dict | None = None,
    use_metadata: bool = False,
) -> dict:
    if use_metadata and metadata_result is not None:
        metadata_status = metadata_result.get("contrast_status", "unknown")
        if metadata_status == "unknown":
            metadata_status = "unclear"
        prompt = CONTRAST_WITH_METADATA_PROMPT_TEMPLATE.format(
            modality=modality or "unknown",
            metadata_status=metadata_status,
            metadata_basis=metadata_result.get("contrast_metadata_basis", ""),
        )
    else:
        prompt = CONTRAST_PROMPT_TEMPLATE
    return _run_medgemma_contrast_with_prompt(montage_image, prompt)
