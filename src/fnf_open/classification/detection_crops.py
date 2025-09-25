"""Generate classification crops from detector predictions."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ..detection.models import build_faster_rcnn
from ..detection.training import load_detection_checkpoint

LOGGER = logging.getLogger(__name__)

AP_CLASS_IDS = {"Left": 1, "Right": 2}
LAT_CLASS_IDS = {"LAT_Neck": 1}


def _ensure_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image.astype(np.float32, copy=False)


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = _ensure_three_channel(image)
    tensor = torch.from_numpy(image.copy())
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor


def _pick_box(
    labels: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    class_id: int,
    score_thr: float,
) -> Optional[np.ndarray]:
    mask = (labels == class_id) & (scores >= score_thr)
    if not np.any(mask):
        return None
    class_scores = scores[mask]
    class_boxes = boxes[mask]
    best_idx = int(np.argmax(class_scores))
    return class_boxes[best_idx]


def _crop_and_resize(image: np.ndarray, box: np.ndarray, size: int = 256) -> Optional[np.ndarray]:
    if box is None:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32, copy=False)


def _load_detection_images_by_fold(pickle_path: Path, view: str) -> Dict[str, Dict[str, np.ndarray]]:
    if pickle_path is None:
        return {}
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)

    if view.upper() == "AP":
        image_key = "Detection_image_AP"
    else:
        image_key = "Detection_image_LAT"

    images_by_fold: Dict[str, Dict[str, np.ndarray]] = {}

    def _build_fold_entry(fold_name: str, fold_data: dict) -> None:
        serials = fold_data.get("Serial")
        images = fold_data.get(image_key)
        if serials is None or images is None:
            raise KeyError(
                f"Detection pickle missing keys for view '{view}': expected 'Serial' and '{image_key}'"
            )
        fold_images: Dict[str, np.ndarray] = {}
        for serial, image in zip(serials, images):
            fold_images[str(serial)] = np.asarray(image)
        images_by_fold[fold_name] = fold_images

    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        # Internal format with explicit folds (e.g. fold1..fold5)
        for fold_name, fold_data in data.items():
            _build_fold_entry(str(fold_name), fold_data)
    elif isinstance(data, dict):
        # Single fold dataset; treat as "all"
        _build_fold_entry("all", data)
    else:
        raise TypeError("Unexpected detection pickle structure; expected a dict")

    return images_by_fold


def _load_detector(checkpoint: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_faster_rcnn(num_classes=num_classes, weights=None)
    model, _ = load_detection_checkpoint(
        model,
        checkpoint_name=checkpoint.name,
        checkpoint_dir=str(checkpoint.parent),
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def build_detection_crops(
    ap_detection_pickle: Path,
    lat_detection_pickle: Optional[Path],
    *,
    ap_checkpoint: Optional[Path] = None,
    lat_checkpoint: Optional[Path] = None,
    ap_checkpoint_template: Optional[str] = None,
    lat_checkpoint_template: Optional[str] = None,
    score_thr: float = 0.3,
    device: torch.device | str = "cpu",
    crop_size: int = 256,
    include_lat: bool = True,
) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    """Return detector-derived crops keyed by patient serial."""

    device_t = torch.device(device)

    ap_images_by_fold = _load_detection_images_by_fold(ap_detection_pickle, "AP")
    if include_lat:
        if lat_detection_pickle is None:
            raise ValueError("LAT detection pickle is required when include_lat is True")
        lat_images_by_fold = _load_detection_images_by_fold(lat_detection_pickle, "LAT")
    else:
        lat_images_by_fold = {}

    if not ap_images_by_fold:
        raise ValueError("AP detection pickle is empty or invalid")
    if include_lat and not lat_images_by_fold:
        raise ValueError("LAT detection pickle is empty or invalid")

    detected: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = {}

    ap_model_cache: Dict[Path, torch.nn.Module] = {}
    lat_model_cache: Dict[Path, torch.nn.Module] = {}

    def _resolve_checkpoint(
        fold: str,
        single: Optional[Path],
        template: Optional[str],
        view: str,
    ) -> Path:
        if template:
            path = Path(template.format(fold=fold))
        elif single is not None:
            path = single
        else:
            raise ValueError(
                f"No checkpoint specified for {view} detections (fold {fold})."
            )
        if not path.exists():
            raise FileNotFoundError(
                f"Detector checkpoint for {view} fold '{fold}' not found: {path}"
            )
        return path

    all_folds = sorted(set(ap_images_by_fold.keys()) | set(lat_images_by_fold.keys()))

    for fold in all_folds:
        ap_images = ap_images_by_fold.get(fold, {})
        lat_images = lat_images_by_fold.get(fold, {}) if include_lat else {}

        ap_checkpoint_path = _resolve_checkpoint(fold, ap_checkpoint, ap_checkpoint_template, "AP")
        lat_checkpoint_path: Optional[Path] = None
        if include_lat:
            lat_checkpoint_path = _resolve_checkpoint(
                fold,
                lat_checkpoint,
                lat_checkpoint_template,
                "LAT",
            )

        if ap_checkpoint_path not in ap_model_cache:
            ap_model_cache[ap_checkpoint_path] = _load_detector(
                ap_checkpoint_path,
                num_classes=len(AP_CLASS_IDS) + 1,
                device=device_t,
            )
        if include_lat and lat_checkpoint_path is not None and lat_checkpoint_path not in lat_model_cache:
            lat_model_cache[lat_checkpoint_path] = _load_detector(
                lat_checkpoint_path,
                num_classes=len(LAT_CLASS_IDS) + 1,
                device=device_t,
            )

        ap_model = ap_model_cache[ap_checkpoint_path]
        lat_model = lat_model_cache.get(lat_checkpoint_path) if include_lat else None

        serials = set(ap_images.keys()) | set(lat_images.keys())
        for serial in sorted(serials):
            ap_image = ap_images.get(serial)
            lat_image = lat_images.get(serial)

            left_crop: Optional[np.ndarray] = None
            right_crop: Optional[np.ndarray] = None
            lat_crop: Optional[np.ndarray] = None

            if ap_image is not None:
                tensor = _image_to_tensor(ap_image)
                with torch.no_grad():
                    output = ap_model([tensor.to(device_t)])[0]
                boxes = output["boxes"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                ap_image_rgb = _ensure_three_channel(ap_image)
                left_box = _pick_box(labels, scores, boxes, AP_CLASS_IDS["Left"], score_thr)
                right_box = _pick_box(labels, scores, boxes, AP_CLASS_IDS["Right"], score_thr)
                left_crop = _crop_and_resize(ap_image_rgb, left_box, crop_size)
                right_crop = _crop_and_resize(ap_image_rgb, right_box, crop_size)
                if left_crop is None or right_crop is None:
                    LOGGER.debug(
                        "Detector missing AP crop(s) for serial %s (fold %s)",
                        serial,
                        fold,
                    )

            if include_lat and lat_image is not None and lat_model is not None:
                tensor = _image_to_tensor(lat_image)
                with torch.no_grad():
                    output = lat_model([tensor.to(device_t)])[0]
                boxes = output["boxes"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                lat_image_rgb = _ensure_three_channel(lat_image)
                lat_box = _pick_box(labels, scores, boxes, LAT_CLASS_IDS["LAT_Neck"], score_thr)
                lat_crop = _crop_and_resize(lat_image_rgb, lat_box, crop_size)
                if lat_crop is None:
                    LOGGER.debug(
                        "Detector missing LAT crop for serial %s (fold %s)",
                        serial,
                        fold,
                    )

            detected[serial] = (left_crop, right_crop, lat_crop)

    return detected
