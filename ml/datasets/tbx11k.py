"""TBX11K manifest builder."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml.datasets.common import inspect_image, read_list
from ml.datasets.manifest import empty_labels, finalize
from ml.datasets.targets import TB, label_column

SOURCE = "tbx11k"
CLASS_FOLDERS = {"health", "sick", "tb"}
UNCERTAIN_CATEGORY = "PulmonaryTuberculosis"


def tbx_root(data_root: Path) -> Path:
    return data_root / "raw" / "tbx11k" / "TBX11K"


def load_tb_categories(json_paths: list[Path]) -> dict[str, set[str]]:
    """file_name -> set of original TB category names from the COCO-style JSON."""
    result: dict[str, set[str]] = {}
    for jp in json_paths:
        data = json.loads(jp.read_text())
        names = {c["id"]: c["name"] for c in data["categories"]}
        files = {img["id"]: img["file_name"].replace("\\", "/") for img in data["images"]}
        for ann in data.get("annotations", []):
            fname = files.get(ann["image_id"])
            if fname is not None:
                result.setdefault(fname, set()).add(names[ann["category_id"]])
    return result


def build_tbx11k(data_root: Path, val_fraction: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, list[dict]]:
    root = tbx_root(data_root)
    train_files = read_list(root / "lists" / "TBX11K_train.txt")
    val_files = read_list(root / "lists" / "TBX11K_val.txt")
    overlap = set(train_files) & set(val_files)
    if overlap:
        raise ValueError(f"TBX11K official train and val overlap: {sorted(overlap)[:5]}")

    json_dir = root / "annotations" / "json"
    categories = load_tb_categories(
        [p for p in (json_dir / "TBX11K_train.json", json_dir / "TBX11K_val.json") if p.exists()])

    rows, skipped = [], []
    entries = [(f, "train") for f in train_files] + [(f, "val") for f in val_files]
    for fname, official in tqdm(entries, desc="TBX11K"):
        image_id = f"{SOURCE}:{fname}"
        folder = fname.split("/")[0]
        if folder not in CLASS_FOLDERS:
            skipped.append({"image_id": image_id, "source": SOURCE, "reason": f"unknown_folder: {folder}"})
            continue
        path = root / "imgs" / fname
        info, reason = inspect_image(path)
        if info is None:
            skipped.append({"image_id": image_id, "source": SOURCE, "reason": reason})
            continue
        cats = categories.get(fname, set())
        if folder != "tb" and cats:
            skipped.append({"image_id": image_id, "source": SOURCE,
                            "reason": f"label_conflict: {folder} image has TB boxes {sorted(cats)}"})
            continue

        labels = empty_labels()
        labels[label_column(TB)] = 1 if folder == "tb" else 0
        rows.append({
            "image_id": image_id, "source": SOURCE,
            "raw_path": path.relative_to(data_root).as_posix(),
            "original_label": folder, "original_subtypes": ";".join(sorted(cats)),
            "uncertain": UNCERTAIN_CATEGORY in cats,
            "annotation_type": "image_label+bbox" if cats else "image_label",
            "label_source": "expert", "official_split": official,
            "split": "test" if official == "val" else "train",
            **info, **labels,
        })

    df = finalize(rows)
    train_idx = df.index[df["official_split"] == "train"]
    if len(train_idx):
        _, val_idx = train_test_split(train_idx, test_size=val_fraction, random_state=seed,
                                      stratify=df.loc[train_idx, "original_label"])
        df.loc[val_idx, "split"] = "val"
    return df, skipped
