"""Preprocessing tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import ml.preprocessing.preprocess as pp
from ml.datasets.build_manifests import build_all
from ml.eda.run_eda import load_manifests
from ml.tests.test_eda_probes import _write_nih_images
from ml.tests.test_manifests import data_root  # noqa: F401  (pytest fixture)


def _noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random((h, w)) * 200 + 30).astype(np.uint8)


def test_trim_pad_resize():
    arr = np.zeros((500, 400), np.uint8)
    arr[40:460, 30:370] = _noise(420, 340)
    out, t = pp.preprocess_array(arr)
    assert out.shape == (320, 320) and out.dtype == np.uint8
    assert (t["crop_x0"], t["crop_y0"], t["crop_x1"], t["crop_y1"]) == (30, 40, 370, 460)
    assert t["square_side"] == 420 and t["pad_left"] == 40 and t["pad_top"] == 0


def test_thin_borders_are_not_trimmed():
    arr = _noise(1024, 1024, seed=7)
    arr[:3, :] = 0
    arr[:, -5:] = 0
    _, t = pp.preprocess_array(arr)
    assert (t["crop_x0"], t["crop_y0"], t["crop_x1"], t["crop_y1"]) == (0, 0, 1024, 1024)
    assert t["pad_left"] == 0 and t["pad_top"] == 0


def test_intermediate_step_harmonizes_resolution():
    big = np.asarray(Image.fromarray(_noise(64, 64, seed=8)).resize((4000, 4000), Image.BICUBIC))
    out, _ = pp.preprocess_array(big)
    two_step = Image.fromarray(big).resize((512, 512), Image.LANCZOS, reducing_gap=3.0)
    expected = np.asarray(two_step.resize((320, 320), Image.LANCZOS, reducing_gap=3.0))
    assert np.array_equal(out, expected)


def test_no_trim_when_content_too_small():
    arr = np.zeros((400, 400), np.uint8)
    arr[150:250, 150:250] = _noise(100, 100)
    _, t = pp.preprocess_array(arr)
    assert (t["crop_x0"], t["crop_y0"], t["crop_x1"], t["crop_y1"]) == (0, 0, 400, 400)


def test_inverted_palette_reads_correctly(tmp_path: Path):
    gray = _noise(64, 64, seed=3)
    idx = 255 - gray
    im = Image.fromarray(idx, mode="L").convert("P")
    im.putpalette([v for i in range(256) for v in (255 - i,) * 3])
    path = tmp_path / "p.png"
    im.save(path)
    assert np.abs(pp.load_gray(path).astype(int) - gray.astype(int)).max() <= 1


def test_box_maps_onto_lesion():
    arr = _noise(1024, 1024, seed=5) // 4
    arr[300:360, 500:580] = 255
    out, t = pp.preprocess_array(arr)
    x, y, w, h = pp.map_box(500, 300, 80, 60, t)
    inside = out[int(y) + 2:int(y + h) - 2, int(x) + 2:int(x + w) - 2]
    assert inside.mean() > 200 and out.mean() < 80


def test_run_source_end_to_end_and_resume(data_root: Path):
    build_all(data_root, ["tbx11k", "shenzhen", "montgomery", "nih"], tbx_val=0.2)
    nih = load_manifests(data_root / "processed" / "manifests")["nih"]
    _write_nih_images(data_root, nih)
    broken = data_root / nih.iloc[0]["raw_path"]
    broken.write_bytes(b"not an image")

    sample = pp.run_source(data_root, "nih", workers=1, limit=9)
    assert (sample["proc_status"] == "ok").sum() >= 1
    full = pp.run_source(data_root, "nih", workers=1)
    assert len(full) == len(nih)
    assert (full["proc_status"] == "failed").sum() == 1
    assert full.loc[full["proc_status"] == "failed", "image_id"].iloc[0] == nih.iloc[0]["image_id"]
    ok = full[full["proc_status"] == "ok"]
    with Image.open(data_root / ok.iloc[0]["processed_path"]) as im:
        assert im.size == (320, 320) and im.mode == "L"
    assert not (data_root / "processed" / "manifests" / "nih_processed.partial.csv").exists()
    assert set(pp.STAT_COLS + pp.TRANSFORM_COLS) <= set(full.columns)

    again = pp.run_source(data_root, "nih", workers=1)
    assert len(again) == len(full)
    assert (again["preprocess_version"] == pp.VERSION).all()

    final = data_root / "processed" / "manifests" / "nih_processed.csv"
    stale = pd.read_csv(final)
    stale["preprocess_version"] = "v0"
    stale.to_csv(final, index=False)
    redone = pp.run_source(data_root, "nih", workers=1)
    assert (redone["preprocess_version"] == pp.VERSION).all()

    boxes = pp.transform_bboxes(data_root, full)
    assert (data_root / "processed" / "manifests" / "nih_bboxes_320.csv").exists()
    assert len(boxes) >= 1

    tbx = pp.run_source(data_root, "tbx11k", workers=1)
    assert tbx["processed_path"].str.contains("images_320/tbx11k/(?:health|sick|tb)/", regex=True).all()
    summary = pp.summarize({"nih": full, "tbx11k": tbx})
    assert summary["datasets"]["nih"]["failed"][0]["proc_status"] == "failed"
