"""Helpers to coalesce dataset column names to a standard English schema."""

from __future__ import annotations
import pandas as pd
from typing import Dict

# Known variants → canonical English names.
# The Korean keys correspond to the original dataset headers and must remain
# unchanged so the mapping aligns with the raw data.
COLUMN_ALIASES: Dict[str, str] = {
    "환자번호": "patient_id",
    "정답label(by CT)": "ct_label",
    "Garden type": "garden_type",
    "Garden_Type": "garden_type",
    "Label_CT": "ct_label",
    "patientID": "patient_id",
}

CANONICAL = {"patient_id", "ct_label", "garden_type"}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {col: COLUMN_ALIASES.get(col, col) for col in df.columns}
    df = df.rename(columns=mapping)
    return df
