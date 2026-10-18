"""Manifest builder tests on synthetic datasets."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.datasets.build_manifests import build_all
from ml.datasets.manifest import validate_manifest
from ml.datasets.nlm_tb import parse_reading
from ml.datasets.targets import LABEL_COLUMNS, NIH_FINDINGS, TB, label_column

TB_COL = label_column(TB)
CATEGORIES = [
    {"id": 1, "name": "ActiveTuberculosis", "supercategory": "Tuberculosis"},
    {"id": 2, "name": "ObsoletePulmonaryTuberculosis", "supercategory": "Tuberculosis"},
    {"id": 3, "name": "PulmonaryTuberculosis", "supercategory": "Tuberculosis"},
]


def _pattern(seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).random((16, 16)) * 255).astype(np.uint8)


def _save(path: Path, seed: int, size: int = 128, sixteen_bit: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(_pattern(seed)).resize((size, size), Image.BICUBIC)
    if sixteen_bit:
        img = Image.fromarray(np.asarray(img, dtype=np.uint16) * 257)
    img.save(path)


def _coco(files: dict[str, list[int]]) -> dict:
    images, anns = [], []
    for i, (f, cats) in enumerate(files.items(), start=1):
        images.append({"id": i, "file_name": f, "width": 128, "height": 128})
        anns += [{"id": len(anns) + 1, "image_id": i, "category_id": c, "bbox": [1, 1, 5, 5]} for c in cats]
    return {"info": {}, "licenses": [], "categories": CATEGORIES, "images": images, "annotations": anns}


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"

    # ---- TBX11K ----
    tbx = root / "raw" / "tbx11k" / "TBX11K"
    (tbx / "lists").mkdir(parents=True)
    (tbx / "annotations" / "json").mkdir(parents=True)
    seed, train, val, tcats, vcats = 1000, [], [], {}, {}
    for folder, n_train, n_val in (("health", 20, 6), ("sick", 20, 6), ("tb", 20, 6)):
        for k in range(n_train + n_val):
            fname = f"{folder}/{folder}{k:04d}.png"
            seed += 1
            _save(tbx / "imgs" / fname, seed)
            (train if k < n_train else val).append(fname)
            if folder == "tb":
                (tcats if k < n_train else vcats)[fname] = [1]
    tcats["tb/tb0001.png"] = [1, 2]                      # active + latent
    vcats["tb/tb0020.png"] = [3]                         # uncertain TB
    tcats["sick/sick0003.png"] = [1]                     # label conflict
    (tbx / "imgs" / "health" / "health0000.png").write_bytes(b"not an image")  # corrupt
    train.append("health/missing.png")                   # missing
    shutil.copy(tbx / "imgs" / "sick" / "sick0020.png", tbx / "imgs" / "sick" / "sick0005.png")  # leak into test
    # A TBX11K training image that is a resized copy of a Shenzhen external-test image.
    _save(tbx / "imgs" / "health" / "health0007.png", seed=5, size=96)
    (tbx / "lists" / "TBX11K_train.txt").write_text("\n".join(train))
    (tbx / "lists" / "TBX11K_val.txt").write_text("\n".join(val))
    (tbx / "annotations" / "json" / "TBX11K_train.json").write_text(json.dumps(_coco({f: tcats.get(f, []) for f in train})))
    (tbx / "annotations" / "json" / "TBX11K_val.json").write_text(json.dumps(_coco({f: vcats.get(f, []) for f in val})))

    # ---- Shenzhen ----
    sz = root / "raw" / "shenzhen"
    readings = {
        "CHNCXR_0001_0": "male 45yrs\nnormal",
        "CHNCXR_0002_0": "female 8month\nnormal",            # pediatric
        "CHNCXR_0003_1": "male 28yrs\nSTB",
        "CHNCXR_0004_1": "Female 33 yrs\nnormal",            # reading disagrees with label
        "CHNCXR_0005_0": "male 60yrs\nnormal",               # same pattern as TBX health0007 (seed 5)
    }
    for i, (stem, text) in enumerate(readings.items(), start=1):
        _save(sz / "CXR_png" / f"{stem}.png", seed=i, size=200, sixteen_bit=(stem == "CHNCXR_0003_1"))
        (sz / "ClinicalReadings").mkdir(parents=True, exist_ok=True)
        (sz / "ClinicalReadings" / f"{stem}.txt").write_text(text)
    _save(sz / "CXR_png" / "notes.png", seed=99)                 # unexpected name

    # ---- Montgomery ----
    mc = root / "raw" / "montgomery"
    for i, (stem, text) in enumerate({
        "MCUCXR_0001_0": "Patient's Sex: F\nPatient's Age: 027Y\nnormal",
        "MCUCXR_0002_1": "Patient's Sex: M\nPatient's Age: 052Y\nbilateral infiltrates",
    }.items(), start=50):
        _save(mc / "CXR_png" / f"{stem}.png", seed=i, size=160)
        (mc / "ClinicalReadings").mkdir(parents=True, exist_ok=True)
        (mc / "ClinicalReadings" / f"{stem}.txt").write_text(text)
    _save(mc / "CXR_png" / "MCUCXR_0003_0.png", seed=60, size=160)  # no reading

    # ---- NIH ----
    nih = root / "raw" / "nih-cxr14"
    nih.mkdir(parents=True)
    header = ("Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Sex,View Position,"
              "OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]")
    rows, train_val, test = [], [], []
    rng = np.random.default_rng(0)
    for pid in range(1, 61):
        for fu in range(3):
            name = f"{pid:08d}_{fu:03d}.png"
            picks = [f for f in NIH_FINDINGS if rng.random() < 0.12]
            label = "|".join(picks) if picks else "No Finding"
            rows.append(f"{name},{label},{fu},{pid},{40 + pid % 30},{'M' if pid % 2 else 'F'},"
                        f"{'PA' if fu % 2 == 0 else 'AP'},2500,2048,0.168,0.168")
            (test if pid > 48 else train_val).append(name)
    rows.append("00000061_000.png,Effusion,0,61,414,M,PA,2500,2048,0.168,0.168")      # implausible age
    train_val.append("00000061_000.png")
    rows.append("00000062_000.png,Cardiomegaly|Alien,0,62,50,F,PA,2500,2048,0.168,0.168")  # unknown label
    train_val.append("00000062_000.png")
    rows.append("00000063_000.png,No Finding|Effusion,0,63,50,F,PA,2500,2048,0.168,0.168")  # contradiction
    train_val.append("00000063_000.png")
    rows.append("00000064_000.png,Mass,0,64,50,F,PA,2500,2048,0.168,0.168")               # in no list
    (nih / "Data_Entry_2017_v2020.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    (nih / "train_val_list.txt").write_text("\n".join(train_val))
    (nih / "test_list.txt").write_text("\n".join(test))
    (nih / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "00000050_000.png,Infiltrate,10.5,20.5,30.0,40.0\n"
        "00000051_001.png,Mass,1,2,3,4\n"
        "99999999_000.png,Mass,1,2,3,4\n")
    return root


def test_parse_reading_formats():
    assert parse_reading("male 28yrs\nSTB") == ("M", 28, "STB")
    assert parse_reading("Patient's Sex: F\nPatient's Age: 027Y\nnormal") == ("F", 27, "normal")
    sex, age, finding = parse_reading("female 8month\nnormal")
    assert sex == "F" and age == pytest.approx(0.67) and finding == "normal"
    # Typos seen in the real Shenzhen / Montgomery readings.
    assert parse_reading("male 42\nnormal") == ("M", 42, "normal")
    assert parse_reading("femal 32yrs\nnormal") == ("F", 32, "normal")
    assert parse_reading("female24yrs\nnormal") == ("F", 24, "normal")
    assert parse_reading("Patient's Sex: O\nPatient's Age: 005Y\nnormal") == ("", 5, "normal")


def test_build_all(data_root: Path):
    summary = build_all(data_root, ["tbx11k", "shenzhen", "montgomery", "nih"], tbx_val=0.2)
    out = data_root / "processed" / "manifests"
    tbx = pd.read_csv(out / "tbx11k_manifest.csv")
    sz = pd.read_csv(out / "shenzhen_manifest.csv")
    mc = pd.read_csv(out / "montgomery_manifest.csv")
    nih = pd.read_csv(out / "nih_manifest.csv", dtype={"patient_id": str})
    skipped = pd.read_csv(out / "skipped.csv")
    reasons = dict(zip(skipped["image_id"], skipped["reason"]))

    # --- TBX11K ---
    assert reasons["tbx11k:health/health0000.png"].startswith("corrupt_image")
    assert reasons["tbx11k:health/missing.png"] == "missing_file"
    assert reasons["tbx11k:sick/sick0003.png"].startswith("label_conflict")
    # Exact copy of a test image removed from train; resized copy of an external image removed too.
    assert reasons["tbx11k:sick/sick0005.png"] == "duplicate_of_image_in_more_important_split"
    assert reasons["tbx11k:health/health0007.png"] == "duplicate_of_image_in_more_important_split"
    assert "shenzhen:CHNCXR_0005_0.png" in set(sz["image_id"])
    assert set(tbx.loc[tbx["official_split"] == "val", "split"]) == {"test"}
    assert set(tbx.loc[tbx["official_split"] == "train", "split"]) == {"train", "val"}
    assert (tbx[TB_COL].isin([0, 1])).all()
    assert tbx[[c for c in LABEL_COLUMNS if c != TB_COL]].isna().all().all()  # NIH labels unknown
    row = tbx.set_index("image_id").loc["tbx11k:tb/tb0020.png"]
    assert bool(row["uncertain"]) and row[TB_COL] == 1
    assert not tbx["raw_path"].str.contains("/extra/").any()

    # --- Shenzhen / Montgomery ---
    assert reasons["shenzhen:notes.png"] == "unexpected_file_name"
    assert set(sz["split"]) == {"external_test"} and set(mc["split"]) == {"external_test"}
    s = sz.set_index("image_id")
    assert s.loc["shenzhen:CHNCXR_0003_1.png", "mode"].startswith("I")          # 16-bit read fine
    assert s.loc["shenzhen:CHNCXR_0003_1.png", TB_COL] == 1
    assert "pediatric" in s.loc["shenzhen:CHNCXR_0002_0.png", "notes"]
    assert "reading_says_normal_but_label_tb" in s.loc["shenzhen:CHNCXR_0004_1.png", "notes"]
    m = mc.set_index("image_id")
    assert m.loc["montgomery:MCUCXR_0001_0.png", "sex"] == "F" and m.loc["montgomery:MCUCXR_0001_0.png", "age"] == 27
    assert "no_reading" in m.loc["montgomery:MCUCXR_0003_0.png", "notes"]

    # --- NIH ---
    assert reasons["nih:00000062_000.png"].startswith("unknown_label")
    assert reasons["nih:00000063_000.png"] == "no_finding_combined_with_findings"
    assert reasons["nih:00000064_000.png"] == "not_in_official_lists"
    assert nih.groupby("patient_id")["split"].nunique().max() == 1             # no patient in two splits
    assert set(nih["split"]) == {"train", "val", "test"}
    assert nih[TB_COL].isna().all()
    n = nih.set_index("image_id")
    assert pd.isna(n.loc["nih:00000061_000.png", "age"])
    assert "implausible_age" in n.loc["nih:00000061_000.png", "notes"]
    no_finding = nih[nih["original_label"] == "No Finding"]
    assert (no_finding[[label_column(f) for f in NIH_FINDINGS]] == 0).all().all()

    boxes = pd.read_csv(out / "nih_bboxes.csv")
    assert set(boxes["image_id"]) == {"nih:00000050_000.png", "nih:00000051_001.png"}  # unknown image dropped
    assert "Infiltration" in set(boxes["finding"])                                     # name fixed

    for df in (tbx, sz, mc):
        validate_manifest(_reload(df))
    assert summary["datasets"]["nih"]["total"] == len(nih)
    assert summary["tb_duplicates_removed"] == 2


def _reload(df: pd.DataFrame) -> pd.DataFrame:
    from ml.datasets.manifest import finalize
    return finalize(df.to_dict("records"))


def test_validate_rejects_patient_leak(data_root: Path):
    build_all(data_root, ["nih"])
    nih = pd.read_csv(data_root / "processed" / "manifests" / "nih_manifest.csv")
    df = _reload(nih)
    test_patient = df.loc[df["split"] == "test", "patient_id"].iloc[0]
    df.loc[df.index[df["split"] == "train"][0], "patient_id"] = test_patient
    with pytest.raises(ValueError, match="more than one split"):
        validate_manifest(df)
