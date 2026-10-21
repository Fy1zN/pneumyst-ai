"""Training pipeline tests on tiny synthetic images (CPU, no pretrained download)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ml.training.augment import AugConfig, affine, augment, random_pad
from ml.training.data import XrayDataset, to_tensor
from ml.training.metrics import banded_auroc, binary_report, threshold_for_sensitivity
from ml.training.train import TrainConfig, compute_pos_weight, fit, masked_bce


def _blob_left(size: int = 320) -> np.ndarray:
    img = np.full((size, size), 60, np.uint8)
    img[120:200, 40:110] = 230  # bright blob on the left side only
    return img


def test_augment_keeps_shape_and_never_flips():
    img = _blob_left()
    cfg = AugConfig(pad_p=1.0, affine_p=1.0)
    for seed in range(50):
        out = augment(img, np.random.default_rng(seed), cfg)
        assert out.shape == img.shape and out.dtype == np.uint8
        ys, xs = np.nonzero(out > out.max() * 0.85)
        assert xs.mean() < img.shape[1] / 2, f"blob moved to the right half (seed {seed})"


def test_random_pad_adds_pure_black_bars():
    img = np.full((320, 320), 128, np.uint8)
    cfg = AugConfig(pad_fill=(0.8, 0.8))
    for seed in range(10):
        out = random_pad(img, np.random.default_rng(seed), cfg)
        black_cols = (out == 0).all(axis=0).sum()
        black_rows = (out == 0).all(axis=1).sum()
        assert black_cols == 64 or black_rows == 64


def test_affine_border_is_black():
    out = affine(np.full((100, 100), 200, np.uint8), np.random.default_rng(0),
                 AugConfig(rotate_deg=7, scale=(0.9, 0.9), shift=0))
    assert out[0, 0] == 0 and out[50, 50] == 200


def test_to_tensor_is_three_channel_normalized():
    x = to_tensor(np.full((32, 32), 255, np.uint8))
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert torch.allclose(x[:, 0, 0], torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]))


def test_pos_weight_and_cap():
    labels = np.array([[1, 0], [0, 0], [0, 0], [0, np.nan], [0, 1]], dtype=float)
    w = compute_pos_weight(labels, cap=2.5).numpy()
    assert w[0] == pytest.approx(2.5)  # 4 neg / 1 pos = 4, capped
    assert w[1] == pytest.approx(2.5)  # 3 neg / 1 pos = 3, capped; NaN ignored


def test_masked_bce_ignores_unknown_labels():
    logits = torch.tensor([[2.0, -1.0]])
    y_known = torch.tensor([[1.0, 0.0]])
    y_partial = torch.tensor([[1.0, float("nan")]])
    w = torch.ones(2)
    only_first = masked_bce(logits[:, :1], y_known[:, :1], w[:1])
    assert masked_bce(logits, y_partial, w).item() == pytest.approx(only_first.item())
    assert not torch.isnan(masked_bce(logits, torch.full((1, 2), float("nan")), w))


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(400)]
    p = np.r_[rng.uniform(0.3, 1, 100), rng.uniform(0, 0.7, 400)]
    t = threshold_for_sensitivity(y, p, 0.9)
    rep = binary_report(y, p, t)
    assert rep["sensitivity"] >= 0.9 and rep["tp"] + rep["fn"] == 100


def test_banded_auroc_detects_a_shortcut():
    # Class is fully explained by the statistic: inside each band AUROC is undefined or ~0.5.
    stat = np.r_[np.linspace(0, 1, 300), np.linspace(2, 3, 300)]
    y = np.r_[np.zeros(300), np.ones(300)]
    rng = np.random.default_rng(1)
    p = stat / 3 + rng.normal(0, 0.01, 600)
    bands = banded_auroc(y, p, stat, n_bands=2)
    assert len(bands) == 2 and bands["auroc"].isna().all()


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    """A 40-image fake tbx11k_processed.csv with 64 px images where TB images are brighter."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        split = "train" if i < 28 else "val"
        tb = int(i % 3 == 0)
        folder = "tb" if tb else ("sick" if i % 3 == 1 else "health")
        rel = f"processed/images_320/tbx11k/{folder}/img{i:03d}.png"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        base = 150 if tb else 90
        Image.fromarray(np.clip(rng.normal(base, 20, (64, 64)), 0, 255).astype(np.uint8)).save(tmp_path / rel)
        rows.append({"image_id": f"tbx11k:{folder}/img{i:03d}.png", "split": split, "original_label": folder,
                     "processed_path": rel, "proc_status": "ok", "label_tuberculosis": tb,
                     "px_mean": float(base), "px_sharpness": 100.0})
    mdir = tmp_path / "processed" / "manifests"
    mdir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(mdir / "tbx11k_processed.csv", index=False)
    return tmp_path


def test_dataset_returns_tensor_and_label(tiny_root: Path):
    df = pd.read_csv(tiny_root / "processed" / "manifests" / "tbx11k_processed.csv")
    ds = XrayDataset(df, tiny_root, ["label_tuberculosis"], AugConfig())
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and y.shape == (1,) and y.item() == 1


def test_fit_runs_saves_and_resumes(tiny_root: Path, tmp_path: Path):
    cfg = TrainConfig(task="tb", epochs=2, frozen_epochs=1, batch_size=8, accum_steps=2, workers=0,
                      pretrained=False, memory_efficient=False, amp=False, patience=5)
    run = tmp_path / "run"
    first = fit(replace(cfg, epochs=1), tiny_root, run_dir=run, device="cpu", log=lambda *_: None)
    assert len(first["history"]) == 1 and first["history"]["frozen"].iloc[0]
    for name in ("last.pt", "best.pt", "history.json", "val_predictions.csv", "config.json"):
        assert (run / name).exists(), name

    second = fit(cfg, tiny_root, run_dir=run, device="cpu", log=lambda *_: None)
    hist = json.loads((run / "history.json").read_text())
    assert [h["epoch"] for h in hist] == [1, 2] and len(second["history"]) == 2
    assert not hist[1]["frozen"]
    preds = pd.read_csv(run / "val_predictions.csv")
    assert {"image_id", "label_tuberculosis", "prob_tuberculosis"} <= set(preds.columns) and len(preds) == 12
    assert "auroc_sick_vs_tb" in hist[0]