"""Manifest schema and validation. Labels: 1, 0, or empty (unknown)."""
from __future__ import annotations

import pandas as pd

from ml.datasets.targets import LABEL_COLUMNS

BASE_COLUMNS = [
    "image_id",           # globally unique: "<source>:<file>"
    "source",             # tbx11k / shenzhen / montgomery / nih
    "raw_path",           # path of the original file relative to DATA_ROOT
    "patient_id",         # "<source>:<id>", empty when the dataset has none
    "age",                # years, empty when unknown
    "sex",                # M / F / empty
    "view",               # PA / AP / empty
    "original_label",     # label exactly as the dataset states it
    "original_subtypes",  # extra original detail (TB categories, reading text)
    "uncertain",          # True if the dataset marks the label as uncertain
    "annotation_type",    # image_label or image_label+bbox
    "label_source",       # expert / nlp_mined / clinical_reading
    "official_split",     # split as published by the dataset authors
    "split",              # ours: train / val / test / external_test
    "width",
    "height",
    "mode",               # PIL mode, e.g. L, RGB, I;16
    "sha256",
    "phash",              # perceptual hash (hex), for near-duplicate checks
    "notes",              # ";"-separated flags, e.g. pediatric, implausible_age
]
MANIFEST_COLUMNS = BASE_COLUMNS + LABEL_COLUMNS
ALLOWED_SPLITS = {"train", "val", "test", "external_test"}


def empty_labels() -> dict:
    """All labels unknown."""
    return {c: pd.NA for c in LABEL_COLUMNS}


def finalize(rows: list[dict]) -> pd.DataFrame:
    """Rows -> DataFrame with canonical columns and dtypes."""
    df = pd.DataFrame(rows)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[MANIFEST_COLUMNS]
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype("Int8")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["uncertain"] = df["uncertain"].fillna(False).astype(bool)
    for col in ("patient_id", "sex", "view", "original_subtypes", "official_split", "sha256", "phash", "notes", "mode"):
        df[col] = df[col].fillna("").astype(str)
    return df


def validate_manifest(df: pd.DataFrame) -> None:
    """Raise ValueError on schema or split-leakage problems."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    if df["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image_id, e.g. {df.loc[df['image_id'].duplicated(), 'image_id'].iloc[0]}")

    bad_splits = set(df["split"]) - ALLOWED_SPLITS
    if bad_splits:
        raise ValueError(f"Unknown split values: {bad_splits}")

    labels = df[LABEL_COLUMNS]
    bad_values = set(pd.unique(labels.stack().dropna().astype(int))) - {0, 1}
    if bad_values:
        raise ValueError(f"Label values must be 0, 1 or empty; found {bad_values}")
    if labels.isna().all(axis=1).any():
        raise ValueError("Some rows have no known label at all")

    with_patient = df[df["patient_id"] != ""]
    if len(with_patient):
        per_patient = with_patient.groupby("patient_id")["split"].nunique()
        if (per_patient > 1).any():
            raise ValueError(f"{int((per_patient > 1).sum())} patients appear in more than one split")

    with_hash = df[df["sha256"] != ""]
    if len(with_hash):
        per_hash = with_hash.groupby("sha256")["split"].nunique()
        if (per_hash > 1).any():
            raise ValueError(f"{int((per_hash > 1).sum())} identical images appear in more than one split")
