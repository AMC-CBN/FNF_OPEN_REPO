"""Datasets and transforms for Faster R-CNN hip detection."""

from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

__all__ = [
    "build_detection_transform",
    "detection_collate",
    "PickleHipDetectionDataset",
]


@dataclass(frozen=True)
class _DetectionRecord:
    serial: str
    image: np.ndarray
    ratio: float
    padding: Tuple[int, int, int, int]
    xml_path: str


def build_detection_transform(image_size: Optional[int] = None) -> A.Compose:
    """Create a deterministic transform pipeline for detection inputs."""

    transforms: List[A.BasicTransform] = []
    if image_size:
        transforms.append(A.LongestMaxSize(max_size=image_size))
        transforms.append(A.PadIfNeeded(image_size, image_size, border_mode=0))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def _convert_voc_xml(
    xml_path: str,
    ratio: float,
    padding: Tuple[int, int, int, int],
    classes: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert VOC-style annotations to detection tensors."""

    tree = ET.parse(xml_path)
    root = tree.getroot()
    xmin_pad, ymin_pad, xmax_pad, ymax_pad = padding
    class_to_idx = {name: idx + 1 for idx, name in enumerate(classes)}  # 0 reserved for background

    labels: List[int] = []
    boxes: List[List[int]] = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name", default="")
        label = class_to_idx.get(class_name)
        if label is None:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", default="0")) + xmin_pad
        ymin = float(bbox.findtext("ymin", default="0")) + ymin_pad
        xmax = float(bbox.findtext("xmax", default="0")) + xmax_pad
        ymax = float(bbox.findtext("ymax", default="0")) + ymax_pad
        boxes.append([
            int(xmin * ratio),
            int(ymin * ratio),
            int(xmax * ratio),
            int(ymax * ratio),
        ])
        labels.append(int(label))

    if not labels:
        return (
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0, 4), dtype=torch.float32),
        )

    return (
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(boxes, dtype=torch.float32),
    )


class PickleHipDetectionDataset(Dataset):
    """Dataset wrapper for detection pickles produced by the preprocessing CLI."""

    def __init__(
        self,
        pickle_path: str | Path,
        fold_keys: Sequence[str],
        classes: Sequence[str],
        transform: Optional[A.Compose] = None,
        *,
        view_key: str = "AP",
    ) -> None:
        self._classes = list(classes)
        self._transform = transform
        self._view_key = view_key.upper()

        path = Path(pickle_path)
        with path.open("rb") as handle:
            data = pickle.load(handle)

        img_key = "Detection_image_AP" if self._view_key == "AP" else "Detection_image_LAT"
        ratio_key = "Ratio_AP_list" if self._view_key == "AP" else "Ratio_LAT_list"
        pad_key = "Pad_AP_list" if self._view_key == "AP" else "Pad_LAT_list"
        xml_key = "Xml_path_AP" if self._view_key == "AP" else "Xml_path_LAT"

        records: List[_DetectionRecord] = []
        for fold in fold_keys:
            fold_data = data[fold]
            for serial, image, ratio, padding, xml in zip(
                fold_data["Serial"],
                fold_data[img_key],
                fold_data[ratio_key],
                fold_data[pad_key],
                fold_data[xml_key],
            ):
                records.append(
                    _DetectionRecord(
                        serial=str(serial),
                        image=np.asarray(image),
                        ratio=float(ratio),
                        padding=tuple(int(x) for x in padding),
                        xml_path=str(xml),
                    )
                )

        self._records = records
        self.collate_fn = detection_collate

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, dict]:
        record = self._records[index]
        image = record.image
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)

        if self._transform is not None:
            tensor = self._transform(image=image)["image"]
        else:
            tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        labels, boxes = _convert_voc_xml(
            record.xml_path,
            record.ratio,
            record.padding,
            self._classes,
        )
        target = {
            "boxes": boxes.to(torch.float32),
            "labels": labels.to(torch.int64),
        }
        return record.serial, tensor, target


def detection_collate(batch):
    """Default collate function for detection datasets."""

    return tuple(zip(*batch))
