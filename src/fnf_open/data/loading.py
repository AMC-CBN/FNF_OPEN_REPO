"""Utilities for loading/saving preprocessed datasets from pickle files."""

from __future__ import annotations
from pathlib import Path
from typing import MutableMapping, Sequence
import pickle
import numpy as np

__all__ = ["load_pickle_dataset", "save_pickle_dataset"]

def load_pickle_dataset(path: Path) -> MutableMapping[str, Sequence[np.ndarray]]:
    """Load a pickled dictionary storing arrays for each patient/sample.

    Parameters
    ----------
    path:
        Path to a `.pkl` file created by the preprocessing scripts.

    Returns
    -------
    dict
        Mapping of keys to NumPy arrays or lists of arrays.
    """
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(data)}")
    return data

def save_pickle_dataset(obj: MutableMapping[str, Sequence[np.ndarray]], path: Path) -> None:
    """Write a dataset dictionary to a pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
