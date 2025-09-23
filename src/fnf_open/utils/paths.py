"""Path utilities for shared model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

MODEL_ROOT = Path("models")
_DETECTION_SUBDIR = "detection"
_CLASSIFICATION_SUBDIR = "classification"

DEFAULT_DETECTION_MODEL_DIR = MODEL_ROOT / _DETECTION_SUBDIR
DEFAULT_CLASSIFICATION_MODEL_DIR = MODEL_ROOT / _CLASSIFICATION_SUBDIR

__all__ = [
    "MODEL_ROOT",
    "DEFAULT_DETECTION_MODEL_DIR",
    "DEFAULT_CLASSIFICATION_MODEL_DIR",
    "ensure_model_root",
    "ensure_detection_model_dir",
    "ensure_classification_model_dir",
    "detection_checkpoint_path",
    "classification_checkpoint_path",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_model_root() -> Path:
    """Ensure that the root directory for persisted models exists."""
    return _ensure_dir(MODEL_ROOT)


def ensure_detection_model_dir() -> Path:
    """Ensure and return the default detection model directory."""
    ensure_model_root()
    return _ensure_dir(DEFAULT_DETECTION_MODEL_DIR)


def ensure_classification_model_dir() -> Path:
    """Ensure and return the default classification model directory."""
    ensure_model_root()
    return _ensure_dir(DEFAULT_CLASSIFICATION_MODEL_DIR)


def _resolve_base(directory: Optional[PathLike], default: Path, ensure_parent: bool) -> Path:
    if directory is None:
        base = default
        if ensure_parent:
            _ensure_dir(base)
    else:
        base = Path(directory)
        if ensure_parent:
            _ensure_dir(base)
    return base


def detection_checkpoint_path(
    filename: str,
    directory: Optional[PathLike] = None,
    *,
    ensure_parent: bool = False,
) -> Path:
    """Return the path to a detection checkpoint under the requested directory."""
    base = _resolve_base(directory, DEFAULT_DETECTION_MODEL_DIR, ensure_parent)
    return base / filename


def classification_checkpoint_path(
    filename: str,
    directory: Optional[PathLike] = None,
    *,
    ensure_parent: bool = False,
) -> Path:
    """Return the path to a classification checkpoint under the requested directory."""
    base = _resolve_base(directory, DEFAULT_CLASSIFICATION_MODEL_DIR, ensure_parent)
    return base / filename
