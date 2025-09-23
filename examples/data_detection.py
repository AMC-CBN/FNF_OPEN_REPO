\
import os
import pickle
from typing import List, Dict, Tuple, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset


def build_default_transform(image_size: Optional[int] = None):
    # Minimal, deterministic transform; users can extend.
    tfs = []
    if image_size:
        tfs.append(A.LongestMaxSize(max_size=image_size))
        tfs.append(A.PadIfNeeded(image_size, image_size, border_mode=0))
    tfs.append(ToTensorV2())
    return A.Compose(tfs)


def convert_voc_xml(xml_path: str, ratio: float, padding: Tuple[int, int, int, int], classes: List[str]):
    """
    Convert VOC-style xml to labels/boxes using ratio + padding adjustment.
    Returns (labels:int64 tensor, boxes:float32 tensor)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    xmin_pad, ymin_pad, xmax_pad, ymax_pad = padding
    class_to_idx = {name: idx + 1 for idx, name in enumerate(classes)}  # background=0

    labels = []
    boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        label = class_to_idx.get(class_name)
        if label is None:
            continue
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text) + xmin_pad
        ymin = float(box.find("ymin").text) + ymin_pad
        xmax = float(box.find("xmax").text) + xmax_pad
        ymax = float(box.find("ymax").text) + ymax_pad
        boxes.append([int(xmin * ratio), int(ymin * ratio), int(xmax * ratio), int(ymax * ratio)])
        labels.append(label)
    if not labels:
        return torch.empty((0,), dtype=torch.int64), torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(labels, dtype=torch.int64), torch.tensor(boxes, dtype=torch.float32)


class PickleFNFAPDataset(Dataset):
    """
    Dataset wrapper for the user's pickle format from the original notebooks.
    Expects keys like:
      foldX -> Serial, Detection_image_AP, Ratio_AP_list, Pad_AP_list, Xml_path_AP
    """
    def __init__(
        self,
        pkl_path: str,
        fold_keys: List[str],
        classes: List[str],
        transform=None,
        view_key: str = "AP",  # "AP" or "LAT"
    ):
        self.classes = classes
        self.transform = transform
        self.view_key = view_key
        with open(pkl_path, "rb") as f:
            self.data_dict = pickle.load(f)

        self.serial = []
        self.images = []
        self.ratios = []
        self.paddings = []
        self.xml_paths = []

        # Construct lists by fold keys
        img_key = "Detection_image_AP" if view_key.upper() == "AP" else "Detection_image_LAT"
        ratio_key = "Ratio_AP_list" if view_key.upper() == "AP" else "Ratio_LAT_list"
        pad_key = "Pad_AP_list" if view_key.upper() == "AP" else "Pad_LAT_list"
        xml_key = "Xml_path_AP" if view_key.upper() == "AP" else "Xml_path_LAT"

        for fk in fold_keys:
            self.serial.extend(self.data_dict[fk]["Serial"])
            self.images.extend(self.data_dict[fk][img_key])
            self.ratios.extend(self.data_dict[fk][ratio_key])
            self.paddings.extend(self.data_dict[fk][pad_key])
            self.xml_paths.extend(self.data_dict[fk][xml_key])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ensure 3-channel
        image = self.images[idx]
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        serial = self.serial[idx]
        ratio = self.ratios[idx]
        padding = tuple(self.paddings[idx])
        xml = self.xml_paths[idx]

        labels, boxes = convert_voc_xml(xml, ratio, padding, self.classes)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        target = {"boxes": boxes.to(torch.float32), "labels": labels.to(torch.int64)}
        return serial, image, target


def detection_collate(batch):
    return tuple(zip(*batch))
