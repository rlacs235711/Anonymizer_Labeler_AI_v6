"""Render DICOM series into representative images for vision-language models."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DEFAULT_CT_WINDOW_LEVEL, DEFAULT_CT_WINDOW_WIDTH, DEFAULT_IMAGE_SIZE, DEFAULT_NUM_SLICES, FULL_SERIES_MONTAGE_TILE_SIZE


def prepare_slice_for_rendering(
    slice_array: np.ndarray,
    modality: str,
    window_level: float = DEFAULT_CT_WINDOW_LEVEL,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> np.ndarray:
    image = slice_array.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    if modality == "CT":
        low, high = window_level - window_width / 2, window_level + window_width / 2
    else:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
    image = np.clip(image, low, high)
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
    return (image * 255).astype(np.uint8)


def _resize_rgb(image: np.ndarray, image_size: int) -> Image.Image:
    pil = Image.fromarray(image, mode="L")
    pil.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), 0)
    canvas.paste(pil, ((image_size - pil.size[0]) // 2, (image_size - pil.size[1]) // 2))
    return canvas.convert("RGB")


def _sample_indices(n: int, k: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(k, dtype=int)
    return np.rint(np.linspace(0, n - 1, k)).astype(int)


def all_slice_indices(n: int) -> np.ndarray:
    return np.arange(max(n, 1), dtype=int)


def _target_slice_count(n: int) -> int:
    if n <= 1:
        return 1
    if n < 80:
        return min(n, 24)
    if n <= 300:
        return min(n, 36)
    return min(n, 48)


def _downsample_for_difference(image: np.ndarray, size: int = 64) -> np.ndarray:
    pil = Image.fromarray(image, mode="L")
    pil.thumbnail((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def select_adaptive_slice_indices(
    series_data: dict,
    max_slices: int = 64,
    window_level: float = DEFAULT_CT_WINDOW_LEVEL,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> np.ndarray:
    """Select broad coverage slices plus slices where anatomy changes most."""
    volume = series_data["volume"]
    modality = series_data.get("metadata", {}).get("Modality", "")
    n = int(volume.shape[0])
    if n <= 1:
        return np.asarray([0], dtype=int)

    target = min(_target_slice_count(n), max_slices, n)
    coverage_count = max(1, int(round(target * 0.7)))
    change_count = max(0, target - coverage_count)
    coverage = set(np.rint(np.linspace(0, n - 1, coverage_count)).astype(int).tolist())

    change_indices: set[int] = set()
    if change_count > 0 and n > 2:
        probe_count = min(n, max(target * 4, 40))
        probe = np.unique(np.rint(np.linspace(0, n - 1, probe_count)).astype(int))
        rendered = [
            _downsample_for_difference(prepare_slice_for_rendering(volume[i], modality, window_level, window_width))
            for i in probe
        ]
        changes = []
        for pos in range(1, len(probe)):
            score = float(np.mean(np.abs(rendered[pos] - rendered[pos - 1])))
            changes.append((score, int(probe[pos])))
        changes.sort(reverse=True)
        for _, index in changes:
            if len(change_indices) >= change_count:
                break
            change_indices.add(index)

    selected = sorted(coverage | change_indices)
    if len(selected) > target:
        positions = np.rint(np.linspace(0, len(selected) - 1, target)).astype(int)
        selected = [selected[pos] for pos in positions]
    return np.asarray(selected, dtype=int)


def _render_indices_as_montages(
    series_data: dict,
    indices: np.ndarray,
    max_tiles_per_montage: int,
    image_size: int,
    window_level: float,
    window_width: float,
) -> list[Image.Image]:
    volume = series_data["volume"]
    modality = series_data.get("metadata", {}).get("Modality", "")
    montages = []
    for start in range(0, len(indices), max_tiles_per_montage):
        chunk = indices[start : start + max_tiles_per_montage]
        tiles = [
            _resize_rgb(prepare_slice_for_rendering(volume[i], modality, window_level, window_width), image_size)
            for i in chunk
        ]
        cols = math.ceil(math.sqrt(len(tiles)))
        rows = math.ceil(len(tiles) / cols)
        montage = Image.new("RGB", (cols * image_size, rows * image_size), (0, 0, 0))
        for tile_index, tile in enumerate(tiles):
            montage.paste(tile, ((tile_index % cols) * image_size, (tile_index // cols) * image_size))
        montages.append(montage)
    return montages


def render_series_montage(
    series_data: dict,
    num_slices: int = DEFAULT_NUM_SLICES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int | None = None,
    window_level: float = DEFAULT_CT_WINDOW_LEVEL,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
    save_path: str | Path | None = None,
) -> Image.Image:
    volume = series_data["volume"]
    modality = series_data.get("metadata", {}).get("Modality", "")
    indices = _sample_indices(volume.shape[0], num_slices)
    tiles = [
        _resize_rgb(prepare_slice_for_rendering(volume[i], modality, window_level, window_width), image_size)
        for i in indices
    ]
    cols = grid_size or math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)
    montage = Image.new("RGB", (cols * image_size, rows * image_size), (0, 0, 0))
    for i, tile in enumerate(tiles):
        montage.paste(tile, ((i % cols) * image_size, (i // cols) * image_size))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        montage.save(save_path)
    return montage


def render_full_series_montage(
    series_data: dict,
    image_size: int = FULL_SERIES_MONTAGE_TILE_SIZE,
    window_level: float = DEFAULT_CT_WINDOW_LEVEL,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
    save_path: str | Path | None = None,
) -> Image.Image:
    volume = series_data["volume"]
    modality = series_data.get("metadata", {}).get("Modality", "")
    indices = all_slice_indices(volume.shape[0])
    tiles = [
        _resize_rgb(prepare_slice_for_rendering(volume[i], modality, window_level, window_width), image_size)
        for i in indices
    ]
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    montage = Image.new("RGB", (cols * image_size, rows * image_size), (0, 0, 0))
    for i, tile in enumerate(tiles):
        montage.paste(tile, ((i % cols) * image_size, (i // cols) * image_size))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        montage.save(save_path)
    return montage


def render_every_n_slice_montages(
    series_data: dict,
    step: int = 3,
    max_tiles_per_montage: int = 64,
    image_size: int = FULL_SERIES_MONTAGE_TILE_SIZE,
    window_level: float = DEFAULT_CT_WINDOW_LEVEL,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> list[Image.Image]:
    """Render every nth slice into manageable montage pages for VLM inference."""
    volume = series_data["volume"]
    modality = series_data.get("metadata", {}).get("Modality", "")
    step = max(int(step), 1)
    indices = np.arange(0, max(volume.shape[0], 1), step, dtype=int)
    if indices.size == 0:
        indices = np.asarray([0], dtype=int)

    return _render_indices_as_montages(series_data, indices, max_tiles_per_montage, image_size, window_level, window_width)


def render_adaptive_series_montages(
    series_data: dict,
    max_slices: int = 64,
    max_tiles_per_montage: int = 64,
    image_size: int = FULL_SERIES_MONTAGE_TILE_SIZE,
    window_level: float = DEFAULT_CT_WINDOW_LEVEL,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> tuple[list[Image.Image], np.ndarray]:
    indices = select_adaptive_slice_indices(
        series_data,
        max_slices=max_slices,
        window_level=window_level,
        window_width=window_width,
    )
    montages = _render_indices_as_montages(
        series_data,
        indices,
        max_tiles_per_montage,
        image_size,
        window_level,
        window_width,
    )
    return montages, indices


def render_representative_slice(series_data: dict, image_size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    volume = series_data["volume"]
    modality = series_data.get("metadata", {}).get("Modality", "")
    image = prepare_slice_for_rendering(volume[volume.shape[0] // 2], modality)
    return _resize_rgb(image, image_size)


def render_single_slice(
    series_data: dict,
    slice_index: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    window_level: float = DEFAULT_CT_WINDOW_LEVEL,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> Image.Image:
    volume = series_data["volume"]
    modality = series_data.get("metadata", {}).get("Modality", "")
    index = int(np.clip(slice_index, 0, volume.shape[0] - 1))
    image = prepare_slice_for_rendering(volume[index], modality, window_level, window_width)
    return _resize_rgb(image, image_size)
