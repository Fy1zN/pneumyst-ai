"""NIH ChestX-ray14 manifest builder (images checked later, during preprocessing)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml.datasets.common import read_list
from ml.datasets.manifest import empty_labels, finalize
from ml.datasets.targets import NIH_FINDINGS, NIH_NO_FINDING, label_column

SOURCE = "nih"
CSV_COLUMNS = ["image_index", "finding_labels", "follow_up", "patient_id", "age", "sex", "view",
               "orig_width", "orig_height", "pixel_spacing_x", "pixel_spacing_y"]
BBOX_NAME_FIX = {"Infiltrate": "Infiltration"}


def nih_root(data_root: Path) -> Path:
    return data_root / "raw" / "nih-cxr14"


def find_labels_csv(root: Path) -> Path:
    for name in ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv"):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"No Data_Entry_2017 CSV in {root}")


def read_entries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 11:
        raise ValueError(f"Expected 11 columns in {path.name}, found {df.shape[1]}: {list(df.columns)}")
    df = df.iloc[:, :11]
    df.columns = CSV_COLUMNS
    return df


def build_nih(data_root: Path, val_fraction: float = 0.10, seed: int = 42
              ) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    root = nih_root(data_root)
    entries = read_entries(find_labels_csv(root))
    train_val = set(read_list(root / "train_val_list.txt"))
    test = set(read_list(root / "test_list.txt"))
    if train_val & test:
        raise ValueError(f"NIH official lists overlap: {sorted(train_val & test)[:5]}")

    bbox_images = set()
    bbox_path = root / "BBox_List_2017.csv"
    if bbox_path.exists():
        bbox_images = set(pd.read_csv(bbox_path).iloc[:, 0].astype(str))

    known = set(NIH_FINDINGS)
    rows, skipped = [], []
    for e in entries.itertuples(index=False):
        name = str(e.image_index)
        image_id = f"{SOURCE}:{name}"
        findings = [f.strip() for f in str(e.finding_labels).split("|") if f.strip()]
        unknown = [f for f in findings if f not in known and f != NIH_NO_FINDING]
        if unknown:
            skipped.append({"image_id": image_id, "source": SOURCE, "reason": f"unknown_label: {unknown}"})
            continue
        if NIH_NO_FINDING in findings and len(findings) > 1:
            skipped.append({"image_id": image_id, "source": SOURCE, "reason": "no_finding_combined_with_findings"})
            continue
        if name in test:
            official = "test"
        elif name in train_val:
            official = "train_val"
        else:
            skipped.append({"image_id": image_id, "source": SOURCE, "reason": "not_in_official_lists"})
            continue

        notes = []
        age = pd.to_numeric(e.age, errors="coerce")
        if pd.notna(age) and not 0 <= age <= 110:
            notes.append(f"implausible_age:{age}")
            age = pd.NA
        sex = str(e.sex).strip().upper() if str(e.sex).strip().upper() in ("M", "F") else ""
        view = str(e.view).strip().upper() if str(e.view).strip().upper() in ("PA", "AP") else ""
        if not view:
            notes.append(f"unknown_view:{e.view}")

        labels = empty_labels()
        for f in NIH_FINDINGS:
            labels[label_column(f)] = 1 if f in findings else 0
        rows.append({
            "image_id": image_id, "source": SOURCE,
            "raw_path": f"raw/nih-cxr14/images/{name}",
            "patient_id": f"{SOURCE}:{int(e.patient_id)}",
            "age": age, "sex": sex, "view": view,
            "original_label": "|".join(findings),
            "original_subtypes": f"follow_up={int(e.follow_up)}",
            "uncertain": False,
            "annotation_type": "image_label+bbox" if name in bbox_images else "image_label",
            "label_source": "nlp_mined", "official_split": official,
            "split": "test" if official == "test" else "train",
            "notes": ";".join(notes), **labels,
        })

    df = finalize(rows)

    # Val = whole patients from official train_val.
    tv = df.index[df["official_split"] == "train_val"]
    if len(tv):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        _, val_pos = next(splitter.split(tv, groups=df.loc[tv, "patient_id"]))
        df.loc[tv[val_pos], "split"] = "val"

    return df, skipped, build_bboxes(bbox_path, df)


def build_bboxes(bbox_path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    cols = ["image_id", "finding", "x", "y", "w", "h", "split", "label_present"]
    if not bbox_path.exists():
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(bbox_path).iloc[:, :6]
    raw.columns = ["image_index", "finding", "x", "y", "w", "h"]
    raw["finding"] = raw["finding"].astype(str).str.strip().replace(BBOX_NAME_FIX)
    raw = raw[raw["finding"].isin(NIH_FINDINGS)].copy()
    raw["image_id"] = SOURCE + ":" + raw["image_index"].astype(str)

    by_id = manifest.set_index("image_id")
    raw = raw[raw["image_id"].isin(by_id.index)].copy()
    raw["split"] = raw["image_id"].map(by_id["split"])
    raw["label_present"] = [
        bool(by_id.at[i, label_column(f)] == 1) for i, f in zip(raw["image_id"], raw["finding"])
    ]
    return raw[cols].reset_index(drop=True)
