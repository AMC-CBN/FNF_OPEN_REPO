"""Shared image processing helpers used across dataset pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut


def _first_value(value: Any) -> float:
    """Return the first element of ``value`` if it is a MultiValue."""

    if isinstance(value, pydicom.multival.MultiValue):
        return float(value[0])
    return float(value)


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Normalise ``image`` to the ``[0, 1]`` range."""

    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value == 0:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def calculate_square_padding(image: np.ndarray) -> Dict[str, int]:
    """Calculate the padding required to make ``image`` square."""

    height, width = image.shape[:2]
    if width > height:
        diff = width - height
        top = diff // 2
        bottom = diff - top
        left = right = 0
    elif height > width:
        diff = height - width
        left = diff // 2
        right = diff - left
        top = bottom = 0
    else:
        top = bottom = left = right = 0

    return {"top": int(top), "bottom": int(bottom), "left": int(left), "right": int(right)}


def normalize_with_padding(
    image: np.ndarray,
    desired_size: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[float, float], Dict[str, int]]:
    """Resize ``image`` preserving the aspect ratio and pad to ``desired_size``."""

    desired_width, desired_height = desired_size
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Input image has invalid dimensions")

    aspect_ratio = width / height
    desired_aspect_ratio = desired_width / desired_height

    if aspect_ratio > desired_aspect_ratio:
        new_width = desired_width
        new_height = max(1, int(round(new_width / aspect_ratio)))
    else:
        new_height = desired_height
        new_width = max(1, int(round(new_height * aspect_ratio)))

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    normalised = min_max_normalize(resized_image)

    top_pad = (desired_height - new_height) // 2
    bottom_pad = desired_height - new_height - top_pad
    left_pad = (desired_width - new_width) // 2
    right_pad = desired_width - new_width - left_pad

    padded_image = cv2.copyMakeBorder(
        normalised,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    resize_ratio = (new_width / width, new_height / height)
    padding = {"top": top_pad, "bottom": bottom_pad, "left": left_pad, "right": right_pad}
    return padded_image.astype(np.float32), resize_ratio, padding


def load_dicom_pixels(dicom_path: str | Path) -> Tuple[pydicom.dataset.FileDataset, np.ndarray]:
    """Load a DICOM file and return the dataset and processed pixel array."""

    path = Path(dicom_path).expanduser().resolve()
    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float32)

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = np.max(pixels) - pixels

    if hasattr(dataset, "RescaleIntercept") and getattr(dataset, "RescaleIntercept") >= 0:
        pixels = apply_modality_lut(pixels, dataset)

    if hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth"):
        window_center = _first_value(dataset.WindowCenter)
        window_width = _first_value(dataset.WindowWidth)
        lower_bound = window_center - (window_width / 2)
        upper_bound = window_center + (window_width / 2)
        pixels = np.clip(pixels, lower_bound, upper_bound)

    return dataset, pixels.astype(np.float32)


__all__ = [
    "calculate_square_padding",
    "load_dicom_pixels",
    "min_max_normalize",
    "normalize_with_padding",
]
