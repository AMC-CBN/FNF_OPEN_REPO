"""Dataset utilities for multi-view femoral neck fracture classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "PreparedClassificationDataset",
    "prepare_classification_dataset",
    "MultiViewDataset",
]


@dataclass(frozen=True)
class PreparedClassificationDataset:
    """Container holding filtered multi-view tensors and labels."""

    serials: List[str]
    ap_left: List[np.ndarray]
    ap_right: List[np.ndarray]
    lateral: List[np.ndarray]
    labels: np.ndarray
    original_labels: np.ndarray
    fold_mapping: Optional[Dict[str, List[int]]] = None

    def __post_init__(self) -> None:  # pragma: no cover - defensive programming
        lengths = {
            len(self.serials),
            len(self.ap_left),
            len(self.ap_right),
            len(self.lateral),
            len(self.labels),
            len(self.original_labels),
        }
        if len(lengths) != 1:
            raise ValueError("All fields in PreparedClassificationDataset must have the same length")

    def __len__(self) -> int:  # pragma: no cover - trivial proxy
        return len(self.labels)


def _detect_key_mapping(dataset: MutableMapping[str, Sequence[np.ndarray]]) -> Dict[str, str]:
    """Return a mapping for serial, views, and labels depending on the pickle schema."""

    candidates = (
        {  # Internal pipeline (`FNF_Classification_data.pkl`)
            "serial": "Serial",
            "ap_left": "Crop_AP_Left_image",
            "ap_right": "Crop_AP_Right_image",
            "lat": "Crop_LAT_image",
            "label": "Garden_Type",
        },
        {  # Legacy external pickles
            "serial": "serial",
            "ap_left": "image_a_L",
            "ap_right": "image_a_R",
            "lat": "image_t",
            "label": "label",
        },
    )

    sample = dataset
    if dataset and isinstance(next(iter(dataset.values())), dict):
        sample = next(iter(dataset.values()))

    for mapping in candidates:
        if all(key in sample for key in mapping.values()):
            return mapping
    raise KeyError(
        "Could not detect dataset schema; expected keys like 'Serial'/'Crop_AP_Left_image' or 'serial'/'image_a_L'."
    )


def _coerce_image(image: np.ndarray) -> np.ndarray:
    """Convert ``image`` to a float32 NumPy array."""

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _append_sample(
    serial: str,
    ap_left: np.ndarray,
    ap_right: np.ndarray,
    lat: np.ndarray,
    label: int,
    original_label: int,
    store: Dict[str, List[np.ndarray]],
    serials: List[str],
    original_labels: List[int],
) -> None:
    serials.append(serial)
    original_labels.append(int(original_label))
    store.setdefault("ap_left", []).append(_coerce_image(ap_left))
    store.setdefault("ap_right", []).append(_coerce_image(ap_right))
    store.setdefault("lat", []).append(_coerce_image(lat))
    store.setdefault("labels", []).append(int(label))


def _iter_fold_samples(
    fold_data: Dict[str, Sequence[np.ndarray]],
    key_mapping: Dict[str, str],
) -> Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
    serials = fold_data[key_mapping["serial"]]
    ap_left_values = fold_data[key_mapping["ap_left"]]
    ap_right_values = fold_data[key_mapping["ap_right"]]
    lat_values = fold_data[key_mapping["lat"]]
    labels = fold_data[key_mapping["label"]]

    return list(
        zip(
            serials,
            ap_left_values,
            ap_right_values,
            lat_values,
            labels,
        )
    )


def prepare_classification_dataset(
    raw_dataset: MutableMapping[str, Sequence[np.ndarray]],
    task: str = "g12_vs_g34",
) -> PreparedClassificationDataset:
    """Filter and remap labels for a specific Garden classification task.

    The function accepts both the internal pipeline pickles (with keys such as
    ``Crop_AP_Left_image``) and the legacy external pickles used in earlier
    notebooks (with keys such as ``image_a_L``).
    """

    if not hasattr(raw_dataset, "keys"):
        raise TypeError(
            "Expected a mapping loaded from a pickle; pass the classification output from fnf-prepare-internal"
        )

    task_normalized = task.lower()
    if task_normalized not in {"g12_vs_g34", "g3_vs_g4"}:
        raise ValueError(f"Unsupported task '{task}'.")

    if not raw_dataset:
        raise ValueError("Classification dataset is empty; ensure preprocessing ran successfully")

    key_mapping = _detect_key_mapping(raw_dataset)

    serials: List[str] = []
    original_labels: List[int] = []
    store: Dict[str, List[np.ndarray]] = {
        "ap_left": [],
        "ap_right": [],
        "lat": [],
        "labels": [],
    }
    fold_mapping: Optional[Dict[str, List[int]]] = None

    def _map_label(label_zero_based: int) -> Optional[int]:
        if task_normalized == "g12_vs_g34":
            if label_zero_based in (0, 1, 2, 3):
                return 0 if label_zero_based <= 1 else 1
            return None
        if task_normalized == "g3_vs_g4":
            if label_zero_based in (2, 3):
                return 0 if label_zero_based == 2 else 1
            return None
        return None

    def _process_sample(
        serial: str,
        left: np.ndarray,
        right: np.ndarray,
        lat: np.ndarray,
        label_value: int,
    ) -> bool:
        if left is None or right is None or lat is None:
            return False
        if isinstance(label_value, np.ndarray):  # pragma: no cover - defensive
            label_int = int(label_value.item())
        else:
            label_int = int(label_value)
        label_zero_based = label_int - 1 if label_int >= 1 else label_int
        mapped = _map_label(label_zero_based)
        if mapped is None:
            return False
        _append_sample(str(serial), left, right, lat, mapped, label_zero_based, store, serials, original_labels)
        return True

    if raw_dataset and isinstance(next(iter(raw_dataset.values())), dict):
        fold_mapping = {}
        running_index = 0
        for fold_name in sorted(raw_dataset.keys()):
            fold = raw_dataset[fold_name]
            if not isinstance(fold, dict):
                raise TypeError("Mixed dataset structure; expected consistent fold dictionaries")
            kept_indices: List[int] = []
            for sample in _iter_fold_samples(fold, key_mapping):
                if _process_sample(*sample):
                    kept_indices.append(running_index)
                    running_index += 1
            fold_mapping[fold_name] = kept_indices
    else:
        dataset = raw_dataset if raw_dataset else {}
        values = zip(
            dataset.get(key_mapping["serial"], []),
            dataset.get(key_mapping["ap_left"], []),
            dataset.get(key_mapping["ap_right"], []),
            dataset.get(key_mapping["lat"], []),
            dataset.get(key_mapping["label"], []),
        )
        for sample in values:
            _process_sample(*sample)

    if not serials:
        raise ValueError("No samples available after filtering; check dataset contents and task selection")

    filtered_dataset = PreparedClassificationDataset(
        serials=serials,
        ap_left=store["ap_left"],
        ap_right=store["ap_right"],
        lateral=store["lat"],
        labels=np.asarray(store["labels"], dtype=int),
        original_labels=np.asarray(original_labels, dtype=int),
        fold_mapping=fold_mapping,
    )

    return filtered_dataset


def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
    """Expand grayscale arrays to three channels expected by timm models."""

    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image.astype(np.float32, copy=False)


class MultiViewDataset(Dataset):
    """Dataset serving three hip views and the corresponding label."""

    def __init__(
        self,
        data: PreparedClassificationDataset,
        transform: A.BasicTransform,
        indices: Sequence[int] | None = None,
    ) -> None:
        self._data = data
        self._transform = transform
        if indices is None:
            self._indices = list(range(len(data)))
        else:
            self._indices = [int(i) for i in indices]

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = int(self._indices[idx])
        ap_left = _ensure_three_channels(self._data.ap_left[real_idx])
        ap_right = _ensure_three_channels(self._data.ap_right[real_idx])
        lat = _ensure_three_channels(self._data.lateral[real_idx])

        transformed = self._transform(
            image=ap_left,
            image_lat=lat,
            image_ap_right=ap_right,
        )

        sample = {
            "ap_left": transformed["image"],
            "ap_right": transformed["image_ap_right"],
            "lat": transformed["image_lat"],
            "label": torch.tensor(self._data.labels[real_idx], dtype=torch.long),
            "original_label": torch.tensor(self._data.original_labels[real_idx], dtype=torch.long),
        }
        return sample

    @property
    def labels(self) -> np.ndarray:
        """Return the labels associated with the selected indices."""

        return self._data.labels[self._indices]

    @property
    def original_labels(self) -> np.ndarray:
        """Return the original Garden labels (zero-based)."""

        return self._data.original_labels[self._indices]
