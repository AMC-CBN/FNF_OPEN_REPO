"""Dataset construction helpers for the CBNU external dataset."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence

import numpy as np
import pandas as pd
from .column_mappings import standardize_columns

from .image_processing import (
    calculate_square_padding,
    load_dicom_pixels,
    normalize_with_padding as _normalize_with_padding,
)

LOGGER = logging.getLogger(__name__)

DESIRED_IMAGE_SIZE = (800, 800)


def _validate_excel_columns(data: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel data: {', '.join(missing)}")


def _load_excel(excel_file_path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    path = Path(excel_file_path).expanduser().resolve()
    data = standardize_columns(pd.read_excel(path, usecols=list(columns)))
    _validate_excel_columns(data, columns)
    return data


def _list_patient_files(split_directory: Path, patient_id: str) -> List[Path]:
    patient_folder = split_directory / patient_id
    if not patient_folder.exists():
        return []
    return sorted(path for path in patient_folder.iterdir() if path.is_file())


def _xml_file(xml_directory: Path, patient_id: str, view: Literal["AP", "LAT"]) -> Path:
    suffix = "AP" if view == "AP" else "LAT"
    return xml_directory / f"{patient_id}_{suffix}.xml"


def _prepare_combined_entry(
    patient_id: str,
    garden_type: Any,
    files: Sequence[Path],
    ap_xml_directory: Path,
    lat_xml_directory: Path,
) -> Dict[str, Any] | None:
    if len(files) < 2:
        return None

    xml_path_a = _xml_file(ap_xml_directory, patient_id, "AP")
    xml_path_t = _xml_file(lat_xml_directory, patient_id, "LAT")

    if not xml_path_a.exists() or not xml_path_t.exists():
        return None

    return {
        "serial": patient_id,
        "image_a": str(files[0]),
        "image_t": str(files[1]),
        "label": garden_type,
        "xml_path_a": str(xml_path_a),
        "xml_path_t": str(xml_path_t),
    }


def build_combined_pickle(
    excel_file_path: str | Path,
    split_directory: str | Path,
    ap_xml_directory: str | Path,
    lat_xml_directory: str | Path,
) -> List[Dict[str, Any]]:
    """Build the combined AP/LAT pickle structure.

    The resulting data structure mirrors the behaviour of the original script but
    focuses solely on data discovery (no image preprocessing).
    """

    data = _load_excel(excel_file_path, ["patient_id", "garden_type"])

    split_path = Path(split_directory).expanduser().resolve()
    ap_xml_path = Path(ap_xml_directory).expanduser().resolve()
    lat_xml_path = Path(lat_xml_directory).expanduser().resolve()

    dataset: List[Dict[str, Any]] = []

    for _, row in data.iterrows():
        patient_id = str(row["patient_id"])
        files = _list_patient_files(split_path, patient_id)
        entry = _prepare_combined_entry(
            patient_id=patient_id,
            garden_type=row["garden_type"],
            files=files,
            ap_xml_directory=ap_xml_path,
            lat_xml_directory=lat_xml_path,
        )
        if entry is not None:
            dataset.append(entry)

    LOGGER.info("Prepared %s combined entries", len(dataset))
    return dataset


def normalize_with_padding(
    image: np.ndarray,
    desired_size: tuple[int, int] = DESIRED_IMAGE_SIZE,
) -> tuple[np.ndarray, tuple[float, float], Dict[str, int]]:
    """Delegate to the shared normalisation helper while keeping the public API."""

    return _normalize_with_padding(image, desired_size)


def preprocess_dicom_image(
    dicom_path: str | Path,
    desired_size: tuple[int, int] = DESIRED_IMAGE_SIZE,
) -> tuple[np.ndarray, float, Dict[str, int]]:
    """Load and preprocess a DICOM image for model input."""

    _, pixels = load_dicom_pixels(dicom_path)

    padded_image, resize_ratio, padding = normalize_with_padding(
        pixels,
        desired_size=desired_size,
    )
    square_padding = calculate_square_padding(pixels)

    if padding["left"] == 0 and padding["right"] == 0:
        effective_ratio = resize_ratio[0]
    else:
        effective_ratio = resize_ratio[1]

    return padded_image.astype(np.float32), float(effective_ratio), square_padding


def build_single_view_pickle(
    excel_file_path: str | Path,
    split_directory: str | Path,
    xml_directory: str | Path,
    view: Literal["AP", "LAT"],
) -> List[Dict[str, Any]]:
    """Build pickle data for a specific projection (AP or LAT)."""

    data = _load_excel(excel_file_path, ["patient_id"])
    split_path = Path(split_directory).expanduser().resolve()
    xml_path = Path(xml_directory).expanduser().resolve()

    file_index = 0 if view == "AP" else 1
    image_key = "image_a" if view == "AP" else "image_t"

    dataset: List[Dict[str, Any]] = []

    for patient_id_value in data["patient_id"]:
        patient_id = str(patient_id_value)
        files = _list_patient_files(split_path, patient_id)
        if len(files) <= file_index:
            continue

        xml_file = _xml_file(xml_path, patient_id, view)
        if not xml_file.exists():
            continue

        preprocessed_image, resize_ratio, padding = preprocess_dicom_image(files[file_index])
        dataset.append(
            {
                "serial": patient_id,
                image_key: preprocessed_image.astype(np.float32),
                "resize_ratio": resize_ratio,
                "padding": padding,
                "xml_path": str(xml_file),
            }
        )

    LOGGER.info("Prepared %s %s entries", len(dataset), view)
    return dataset


def save_pickle(data: Iterable[Dict[str, Any]], output_path: str | Path) -> Path:
    """Serialise ``data`` to ``output_path`` using pickle."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(list(data), fh)
    LOGGER.info("Saved pickle file to %s", path)
    return path


__all__ = [
    "build_combined_pickle",
    "build_single_view_pickle",
    "calculate_square_padding",
    "normalize_with_padding",
    "preprocess_dicom_image",
    "save_pickle",
]
