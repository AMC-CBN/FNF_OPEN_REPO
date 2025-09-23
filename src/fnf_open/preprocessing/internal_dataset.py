"""Dataset preparation helpers for the internal hospital dataset.

The original implementation of this pipeline lived in a monolithic script that
was difficult to test and reason about.  This module provides a structured
alternative with small, composable functions that cover image preprocessing,
annotation parsing and pickle serialisation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from .column_mappings import standardize_columns
import pydicom
from xml.etree import ElementTree as ET

from .image_processing import (
    calculate_square_padding as _calculate_square_padding_dict,
    load_dicom_pixels as _shared_load_dicom_pixels,
    min_max_normalize as _shared_min_max_normalize,
    normalize_with_padding as _shared_normalize_with_padding,
)

LOGGER = logging.getLogger(__name__)

DESIRED_DETECTION_SIZE = (800, 800)
CROP_SIZE = (256, 256)


#: Serial numbers that require a 180 degree rotation for the AP projection.
AP_ROTATE_180_SERIALS = frozenset(
    {
        7,
        10,
        51,
        97,
        127,
        133,
        181,
        182,
        213,
        216,
        233,
        243,
        259,
        268,
        342,
        366,
        431,
        443,
        444,
        484,
        487,
        495,
        499,
        533,
        534,
        574,
        598,
        624,
        628,
        647,
        662,
        683,
        688,
        692,
        700,
        704,
        720,
        754,
        760,
        761,
        774,
        786,
        795,
        804,
        806,
        869,
        888,
        904,
        913,
        927,
        945,
        961,
        983,
        992,
        997,
        1003,
        1039,
        1112,
        1122,
        1124,
        1134,
        1149,
        1151,
        1174,
        1226,
        1240,
        1259,
        1260,
        1265,
        1276,
        1278,
        1300,
        1310,
        1329,
        1339,
        1343,
        1356,
        1414,
        1429,
        1436,
        1451,
        1515,
        1543,
        1562,
        1581,
        1594,
        1631,
        1650,
        1682,
        1715,
        1724,
        1768,
        1790,
        1827,
        1833,
        1894,
        1909,
        1923,
        1927,
        1958,
        2012,
        2015,
        2017,
        2040,
        2041,
        2044,
        2045,
    }
)


#: Serial numbers that require a 90 degree clockwise rotation for the AP
#: projection.
AP_ROTATE_CLOCKWISE_SERIALS = frozenset({51, 182, 243, 259, 443, 598, 628, 804, 1276, 1543, 1562, 2012})


#: Serial numbers that require a 180 degree rotation for the LAT projection.
LAT_ROTATE_180_SERIALS = frozenset(
    {
        10,
        13,
        16,
        24,
        40,
        43,
        45,
        51,
        56,
        59,
        61,
        70,
        76,
        77,
        83,
        96,
        106,
        108,
        123,
        126,
        127,
        144,
        149,
        160,
        162,
        164,
        167,
        169,
        173,
        183,
        187,
        188,
        191,
        192,
        231,
        236,
        239,
        241,
        245,
        250,
        252,
        254,
        256,
        263,
        267,
        270,
        272,
        277,
        283,
        297,
        310,
        314,
        317,
        323,
        334,
        342,
        354,
        364,
        366,
        367,
        374,
        378,
        386,
        388,
        389,
        394,
        403,
        414,
        416,
        419,
        424,
        432,
        438,
        441,
        442,
        451,
        452,
        461,
        466,
        467,
        472,
        473,
        483,
        485,
        494,
        495,
        496,
        505,
        514,
        525,
        539,
        547,
        548,
        553,
        554,
        559,
        575,
        587,
        600,
        608,
        614,
        622,
        624,
        629,
        630,
        632,
        634,
        641,
        643,
        644,
        652,
        658,
        662,
        669,
        673,
        674,
        685,
        692,
        699,
        701,
        706,
        714,
        717,
        720,
        731,
        738,
        742,
        744,
        755,
        766,
        768,
        772,
        774,
        775,
        776,
        781,
        786,
        793,
        805,
        813,
        816,
        817,
        820,
        822,
        825,
        826,
        847,
        850,
        855,
        858,
        862,
        902,
        903,
        911,
        912,
        917,
        919,
        920,
        927,
        934,
        941,
        942,
        945,
        949,
        952,
        963,
        965,
        968,
        970,
        983,
        985,
        992,
        997,
        1002,
        1003,
        1006,
        1017,
        1018,
        1021,
        1027,
        1036,
        1037,
        1039,
        1048,
        1051,
        1060,
        1068,
        1083,
        1087,
        1104,
        1115,
        1119,
        1132,
        1136,
        1143,
        1149,
        1151,
        1153,
        1161,
        1175,
        1181,
        1183,
        1184,
        1188,
        1189,
        1198,
        1199,
        1209,
        1211,
        1213,
        1226,
        1231,
        1237,
        1238,
        1240,
        1243,
        1247,
        1258,
        1260,
        1261,
        1269,
        1274,
        1276,
        1277,
        1281,
        1290,
        1291,
        1292,
        1303,
        1310,
        1324,
        1329,
        1332,
        1339,
        1342,
        1343,
        1346,
        1350,
        1352,
        1359,
        1362,
        1366,
        1367,
        1370,
        1376,
        1386,
        1387,
        1395,
        1398,
        1403,
        1404,
        1408,
        1411,
        1423,
        1428,
        1436,
        1438,
        1439,
        1445,
        1449,
        1451,
        1456,
        1461,
        1468,
        1472,
        1474,
        1479,
        1485,
        1488,
        1508,
        1511,
        1515,
        1516,
        1517,
        1526,
        1542,
        1544,
        1551,
        1555,
        1558,
        1566,
        1577,
        1581,
        1582,
        1590,
        1594,
        1602,
        1604,
        1607,
        1610,
        1617,
        1636,
        1640,
        1643,
        1647,
        1652,
        1653,
        1656,
        1660,
        1665,
        1670,
        1678,
        1681,
        1686,
        1688,
        1690,
        1693,
        1694,
        1707,
        1711,
        1717,
        1720,
        1721,
        1728,
        1733,
        1752,
        1753,
        1763,
        1764,
        1771,
        1776,
        1779,
        1786,
        1792,
        1808,
        1816,
        1825,
        1842,
        1855,
        1856,
        1860,
        1869,
        1871,
        1873,
        1877,
        1880,
        1891,
        1893,
        1894,
        1901,
        1905,
        1913,
        1917,
        1929,
        1931,
        1932,
        1943,
        1950,
        1956,
        1958,
        1970,
        1993,
        2032,
        2036,
        2040,
        2044,
        2126,
        2131,
    }
)


LAT_ROTATE_COUNTERCLOCKWISE_SERIALS = frozenset({585, 1334})
LAT_ROTATE_CLOCKWISE_SERIALS = frozenset({120, 642, 780, 996, 2028, 2101, 2252})
LAT_ROTATE_MINUS_30_SERIALS = frozenset({2269})


@dataclass(frozen=True)
class Padding:
    """Amount of padding required for each image side."""

    left: int
    right: int
    top: int
    bottom: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return the padding as ``(left, bottom, right, top)``."""

        return (int(self.left), int(self.bottom), int(self.right), int(self.top))


@dataclass
class ProjectionResult:
    """Preprocessing output for a single projection (AP or LAT)."""

    original_image: np.ndarray
    detection_image: np.ndarray
    resize_ratio: float
    padding: Padding


@dataclass
class PatientRecord:
    """Aggregated data for a single patient serial."""

    serial: str
    garden_type: int
    dicom_ap_path: Path
    dicom_lat_path: Path
    ap_result: ProjectionResult
    lat_result: ProjectionResult
    ap_left_crop: Optional[np.ndarray]
    ap_right_crop: Optional[np.ndarray]
    lat_crop: Optional[np.ndarray]
    ap_xml_path: Path
    lat_xml_path: Path


def _min_max_normalise(image: np.ndarray) -> np.ndarray:
    """Normalise ``image`` to the ``[0, 1]`` range."""

    return _shared_min_max_normalize(image)


def _calculate_padding(image: np.ndarray) -> Padding:
    """Calculate the padding required to make ``image`` square."""

    padding = _calculate_square_padding_dict(image)
    return Padding(
        left=int(padding["left"]),
        right=int(padding["right"]),
        top=int(padding["top"]),
        bottom=int(padding["bottom"]),
    )


def _normalize_with_padding(
    image: np.ndarray,
    desired_size: Tuple[int, int] = DESIRED_DETECTION_SIZE,
) -> Tuple[np.ndarray, float]:
    """Resize ``image`` while preserving its aspect ratio and pad to ``desired_size``."""

    padded_image, resize_ratio, padding = _shared_normalize_with_padding(image, desired_size)

    if padding["top"] == 0 and padding["bottom"] == 0:
        effective_ratio = resize_ratio[1]
    else:
        effective_ratio = resize_ratio[0]

    return padded_image.astype(np.float32), float(effective_ratio)


def _load_dicom_pixels(dicom_path: Path) -> Tuple[pydicom.dataset.FileDataset, np.ndarray]:
    """Load the DICOM dataset and return the processed pixel array."""

    return _shared_load_dicom_pixels(dicom_path)


def _rotate_if_tall(image: np.ndarray) -> np.ndarray:
    """Rotate images that are significantly taller than they are wide."""

    height, width = image.shape[:2]
    if height > width + 100:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``image`` by ``angle`` degrees around its centre without resizing."""

    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180 or angle == -180:
        return cv2.rotate(image, cv2.ROTATE_180)

    height, width = image.shape[:2]
    centre = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)


def _apply_ap_rotation(serial: int, image: np.ndarray) -> np.ndarray:
    if serial in AP_ROTATE_180_SERIALS:
        return _rotate_image(image, 180)
    if serial in AP_ROTATE_CLOCKWISE_SERIALS:
        return _rotate_image(image, 90)
    return image


def _apply_lat_rotation(serial: int, image: np.ndarray) -> np.ndarray:
    if serial in LAT_ROTATE_180_SERIALS:
        return _rotate_image(image, 180)
    if serial in LAT_ROTATE_COUNTERCLOCKWISE_SERIALS:
        return _rotate_image(image, -90)
    if serial in LAT_ROTATE_CLOCKWISE_SERIALS:
        return _rotate_image(image, 90)
    if serial in LAT_ROTATE_MINUS_30_SERIALS:
        return _rotate_image(image, -30)
    return image


def _preprocess_projection(
    serial: int,
    dicom_path: Path,
    rotation_fn,
    desired_size: Tuple[int, int],
) -> ProjectionResult:
    """Load a DICOM file, apply rotations and produce detection imagery."""

    _, pixels = _load_dicom_pixels(dicom_path)
    pixels = _rotate_if_tall(pixels)
    pixels = rotation_fn(serial, pixels)

    original_image = pixels.astype(np.float32)
    detection_image, resize_ratio = _normalize_with_padding(original_image, desired_size)
    padding = _calculate_padding(original_image)

    return ProjectionResult(
        original_image=original_image,
        detection_image=detection_image.astype(np.float32),
        resize_ratio=float(resize_ratio),
        padding=padding,
    )


def preprocess_ap_image(serial: int, dicom_path: Path, desired_size: Tuple[int, int]) -> ProjectionResult:
    """Preprocess the AP projection for ``serial``."""

    return _preprocess_projection(serial, dicom_path, _apply_ap_rotation, desired_size)


def preprocess_lat_image(serial: int, dicom_path: Path, desired_size: Tuple[int, int]) -> ProjectionResult:
    """Preprocess the LAT projection for ``serial``."""

    return _preprocess_projection(serial, dicom_path, _apply_lat_rotation, desired_size)


def _parse_bbox(obj: ET.Element, image_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Extract a bounding box from an XML object element."""

    box = obj.find("bndbox")
    if box is None:
        return None

    try:
        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))
    except ValueError:
        return None

    height, width = image_shape
    xmin = int(np.clip(xmin, 0, width))
    xmax = int(np.clip(xmax, 0, width))
    ymin = int(np.clip(ymin, 0, height))
    ymax = int(np.clip(ymax, 0, height))

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax


def _crop_region(image: np.ndarray, bbox: Tuple[int, int, int, int], size: Tuple[int, int]) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    cropped = image[ymin:ymax, xmin:xmax]
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_LINEAR)
    return _min_max_normalise(resized)


def crop_ap_regions(
    image: np.ndarray,
    xml_path: Path,
    crop_size: Tuple[int, int] = CROP_SIZE,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract the left and right femoral neck crops from an AP image."""

    if not xml_path.exists():
        raise FileNotFoundError(f"AP XML annotation not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    left_crop: Optional[np.ndarray] = None
    right_crop: Optional[np.ndarray] = None

    for obj in root.findall("object"):
        class_name = obj.findtext("name", default="").strip()
        bbox = _parse_bbox(obj, image.shape[:2])
        if bbox is None:
            continue

        try:
            crop = _crop_region(image, bbox, crop_size)
        except cv2.error:
            continue

        if class_name == "Left":
            left_crop = crop
        elif class_name == "Right":
            right_crop = crop

    return left_crop, right_crop


def crop_lat_region(
    image: np.ndarray,
    xml_path: Path,
    crop_size: Tuple[int, int] = CROP_SIZE,
) -> Optional[np.ndarray]:
    """Extract the femoral neck crop from a LAT image."""

    if not xml_path.exists():
        raise FileNotFoundError(f"LAT XML annotation not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        class_name = obj.findtext("name", default="").strip()
        if class_name != "LAT_Neck":
            continue

        bbox = _parse_bbox(obj, image.shape[:2])
        if bbox is None:
            continue

        try:
            return _crop_region(image, bbox, crop_size)
        except cv2.error:
            continue

    return None


def _normalise_serial(value: Any) -> str:
    """Normalise the serial number extracted from the Excel sheet."""

    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        raise ValueError("Serial value is missing")
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def load_internal_metadata(excel_file_path: Path) -> pd.DataFrame:
    """Load and filter the internal dataset metadata from ``excel_file_path``."""

    raw = pd.read_excel(excel_file_path, usecols=[0, 7, 9, 11], header=None)
    raw.columns = raw.iloc[0]
    data = raw.iloc[1:].copy()
    data = standardize_columns(data)

    required_columns = {"serial No.", "exclusion", "ct_label"}
    missing = required_columns - set(data.columns)
    if missing:
        raise KeyError(
            "Missing required columns in internal label sheet: "
            + ", ".join(sorted(missing))
        )

    data = data[data["exclusion"] == 0]

    data["serial No."] = data["serial No."].apply(_normalise_serial)
    data["ct_label"] = data["ct_label"].astype(float)
    data["Garden_Type"] = data["ct_label"].astype(int) - 1

    return data[["serial No.", "Garden_Type"]].rename(columns={"serial No.": "serial"})


def _build_patient_paths(
    serial: str,
    raw_directory: Path,
    ap_xml_directory: Path,
    lat_xml_directory: Path,
) -> Tuple[Path, Path, Path, Path]:
    folder_path = raw_directory / serial
    ap_dicom = folder_path / f"{serial}a000.dcm"
    lat_dicom = folder_path / f"{serial}t000.dcm"
    ap_xml = ap_xml_directory / f"{serial}a.xml"
    lat_xml = lat_xml_directory / f"{serial}t.xml"
    return ap_dicom, lat_dicom, ap_xml, lat_xml


def _prepare_patient_record(
    serial: str,
    garden_type: int,
    raw_directory: Path,
    ap_xml_directory: Path,
    lat_xml_directory: Path,
    desired_detection_size: Tuple[int, int],
    crop_size: Tuple[int, int],
) -> PatientRecord:
    ap_dicom_path, lat_dicom_path, ap_xml_path, lat_xml_path = _build_patient_paths(
        serial, raw_directory, ap_xml_directory, lat_xml_directory
    )

    missing_paths = [path for path in (ap_dicom_path, lat_dicom_path, ap_xml_path, lat_xml_path) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(str(path) for path in missing_paths)
        )

    serial_int = int(serial)
    ap_result = preprocess_ap_image(serial_int, ap_dicom_path, desired_detection_size)
    lat_result = preprocess_lat_image(serial_int, lat_dicom_path, desired_detection_size)

    ap_left_crop, ap_right_crop = crop_ap_regions(ap_result.original_image, ap_xml_path, crop_size)
    lat_crop = crop_lat_region(lat_result.original_image, lat_xml_path, crop_size)

    return PatientRecord(
        serial=serial,
        garden_type=int(garden_type),
        dicom_ap_path=ap_dicom_path,
        dicom_lat_path=lat_dicom_path,
        ap_result=ap_result,
        lat_result=lat_result,
        ap_left_crop=ap_left_crop,
        ap_right_crop=ap_right_crop,
        lat_crop=lat_crop,
        ap_xml_path=ap_xml_path,
        lat_xml_path=lat_xml_path,
    )


def process_internal_dataset(
    excel_file_path: Path,
    raw_directory: Path,
    ap_xml_directory: Path,
    lat_xml_directory: Path,
    desired_detection_size: Tuple[int, int] = DESIRED_DETECTION_SIZE,
    crop_size: Tuple[int, int] = CROP_SIZE,
) -> List[PatientRecord]:
    """Process all patients defined in the Excel metadata file."""

    metadata = load_internal_metadata(excel_file_path)
    total_records = len(metadata)
    records: List[PatientRecord] = []

    for index, (serial, label) in enumerate(
        metadata.itertuples(index=False), start=1
    ):
        LOGGER.info("Processing serial %s (%d/%d)", serial, index, total_records)
        try:
            record = _prepare_patient_record(
                serial=serial,
                garden_type=label,
                raw_directory=raw_directory,
                ap_xml_directory=ap_xml_directory,
                lat_xml_directory=lat_xml_directory,
                desired_detection_size=desired_detection_size,
                crop_size=crop_size,
            )
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping serial %s due to missing files: %s", serial, exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive logging for unexpected failures
            LOGGER.exception("Error processing serial %s: %s", serial, exc)
            continue

        records.append(record)

    LOGGER.info("Processed %s patient records", len(records))
    return records


def _create_process_lists(records: Sequence[PatientRecord]) -> Dict[str, Dict[str, List[Any]]]:
    """Create per-process data dictionaries mirroring the original script."""

    serials = [record.serial for record in records]
    garden_types = [record.garden_type for record in records]

    return {
        "Paper": {
            "Serial": serials,
            "Dicom_image_path_AP": [str(record.dicom_ap_path) for record in records],
            "Dicom_image_path_LAT": [str(record.dicom_lat_path) for record in records],
            "Garden_Type": garden_types,
        },
        "AP_Detection": {
            "Serial": serials,
            "Detection_image_AP": [record.ap_result.detection_image for record in records],
            "Ratio_AP_list": [record.ap_result.resize_ratio for record in records],
            "Pad_AP_list": [record.ap_result.padding.as_tuple() for record in records],
            "Xml_path_AP": [str(record.ap_xml_path) for record in records],
        },
        "LAT_Detection": {
            "Serial": serials,
            "Detection_image_LAT": [record.lat_result.detection_image for record in records],
            "Ratio_LAT_list": [record.lat_result.resize_ratio for record in records],
            "Pad_LAT_list": [record.lat_result.padding.as_tuple() for record in records],
            "Xml_path_LAT": [str(record.lat_xml_path) for record in records],
        },
        "Classification": {
            "Serial": serials,
            "Crop_AP_Right_image": [record.ap_right_crop for record in records],
            "Crop_AP_Left_image": [record.ap_left_crop for record in records],
            "Crop_LAT_image": [record.lat_crop for record in records],
            "Garden_Type": garden_types,
        },
    }


def _stratified_fold_indices(labels: Sequence[int], folds: int = 5, seed: int = 42) -> Dict[str, List[int]]:
    """Distribute indices across ``folds`` while preserving class balance."""

    if folds <= 0:
        raise ValueError("Number of folds must be positive")

    rng = np.random.default_rng(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(index)

    for indices in label_to_indices.values():
        rng.shuffle(indices)

    fold_indices: Dict[str, List[int]] = {f"fold{i + 1}": [] for i in range(folds)}
    for indices in label_to_indices.values():
        for position, index in enumerate(indices):
            fold_name = f"fold{(position % folds) + 1}"
            fold_indices[fold_name].append(index)

    return fold_indices


def _build_fold_data(process_data: Dict[str, List[Any]], folds: Dict[str, List[int]]) -> Dict[str, Dict[str, List[Any]]]:
    """Slice ``process_data`` into folds using ``folds`` index mapping."""

    fold_data: Dict[str, Dict[str, List[Any]]] = {}
    for fold_name, indices in folds.items():
        fold_data[fold_name] = {
            key: [values[i] for i in indices]
            for key, values in process_data.items()
        }
    return fold_data


def build_internal_pickles(
    excel_file_path: Path,
    raw_directory: Path,
    ap_xml_directory: Path,
    lat_xml_directory: Path,
    output_directory: Path,
    desired_detection_size: Tuple[int, int] = DESIRED_DETECTION_SIZE,
    crop_size: Tuple[int, int] = CROP_SIZE,
) -> Dict[str, Path]:
    """Process the dataset and save per-process pickle files.

    Returns a mapping from process name to the corresponding pickle path.
    """

    records = process_internal_dataset(
        excel_file_path=excel_file_path,
        raw_directory=raw_directory,
        ap_xml_directory=ap_xml_directory,
        lat_xml_directory=lat_xml_directory,
        desired_detection_size=desired_detection_size,
        crop_size=crop_size,
    )

    if not records:
        raise ValueError("No patient records were processed; please check the input paths")

    process_lists = _create_process_lists(records)
    fold_indices = _stratified_fold_indices([record.garden_type for record in records])

    output_directory.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}
    for process_name, process_data in process_lists.items():
        fold_data = _build_fold_data(process_data, fold_indices)
        pickle_path = output_directory / f"FNF_{process_name}_data.pkl"
        with pickle_path.open("wb") as handle:
            pickle.dump(fold_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("%s data saved to %s", process_name, pickle_path)
        saved_paths[process_name] = pickle_path

    return saved_paths


__all__ = [
    "CROP_SIZE",
    "DESIRED_DETECTION_SIZE",
    "build_internal_pickles",
    "crop_ap_regions",
    "crop_lat_region",
    "load_internal_metadata",
    "process_internal_dataset",
    "preprocess_ap_image",
    "preprocess_lat_image",
]
