"""Probe, normalization and NIH duplicate-search tests."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from PIL import Image

import ml.eda.probes as probes
from ml.datasets.build_manifests import build_all
from ml.eda.run_eda import load_manifests
from ml.tests.test_manifests import data_root  # noqa: F401  (pytest fixture)


def test_lsh_matches_brute_force():
    rng = np.random.default_rng(0)
    base = [rng.integers(0, 256, 32, dtype=np.uint8) for _ in range(300)]
    hashes = [b.tobytes().hex() for b in base]
    for d in range(0, 9):
        v = base[d].copy()
        for bit in rng.choice(256, d, replace=False):
            v[bit // 8] ^= 1 << (bit % 8)
        hashes.append(v.tobytes().hex())
    packed = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hashes])
    brute = {(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))
             if int(probes._POPCOUNT[packed[i] ^ packed[j]].sum()) <= 6}
    found, skipped = probes.lsh_pairs(hashes, max_distance=6)
    assert skipped == 0
    assert {(i, j) for i, j, _ in found} == brute
    assert len(brute) == 7  # planted distances 0..6


def _write_nih_images(root: Path, nih: pd.DataFrame) -> None:
    for k, p in enumerate(nih["raw_path"]):
        path = root / p
        path.parent.mkdir(parents=True, exist_ok=True)
        small = (np.random.default_rng(10_000 + k).random((16, 16)) * 255).astype(np.uint8)
        Image.fromarray(small).resize((128, 128), Image.BICUBIC).save(path)


def test_nih_duplicates_and_probes(data_root: Path, tmp_path: Path):
    build_all(data_root, ["tbx11k", "shenzhen", "montgomery", "nih"], tbx_val=0.2)
    dfs = load_manifests(data_root / "processed" / "manifests")
    nih = dfs["nih"]
    _write_nih_images(data_root, nih)

    test_row = nih[nih["split"] == "test"].iloc[0]
    train_rows = nih[(nih["split"] == "train") & (nih["patient_id"] != test_row["patient_id"])]
    leak_row = train_rows.iloc[0]
    shutil.copy(data_root / test_row["raw_path"], data_root / leak_row["raw_path"])
    same = nih[nih["patient_id"] == train_rows.iloc[5]["patient_id"]]
    shutil.copy(data_root / same.iloc[0]["raw_path"], data_root / same.iloc[1]["raw_path"])

    hashes = probes.nih_hashes(nih, data_root, tmp_path / "nih_hashes.csv", workers=1)
    assert (tmp_path / "nih_hashes.csv").exists() and (hashes["phash"] != "").all()
    report, summary = probes.nih_duplicates(nih, hashes, tmp_path / "nih_duplicates.csv")
    leaks = report[report["kind"].str.startswith("LEAK")]
    assert {leak_row["image_id"], test_row["image_id"]} == set(leaks.iloc[0][["image_a", "image_b"]])
    assert (report["kind"] == "same patient").any()
    assert summary["oversized_buckets_skipped"] == 0

    out, local = tmp_path / "out", tmp_path / "local"
    out.mkdir()
    local.mkdir()
    probes.duplicate_grid(report, nih, data_root, local / "nih_leak_pairs.png")
    assert (local / "nih_leak_pairs.png").exists()

    stats, aucs = probes.tbx_class_probe(dfs["tbx11k"], data_root, out, n_per_class=15, methods=tuple(probes.NORMALIZATIONS))
    assert set(stats.index) == {"health", "sick", "tb"}
    assert aucs.shape == (3, 3)

    summary = build_all(data_root, ["nih"], hash_nih=True, workers=1)
    rebuilt = pd.read_csv(data_root / "processed" / "manifests" / "nih_manifest.csv")
    skipped = pd.read_csv(data_root / "processed" / "manifests" / "skipped.csv")
    reasons = dict(zip(skipped["image_id"], skipped["reason"]))
    assert reasons[leak_row["image_id"]] == "duplicate_cross_split"
    assert test_row["image_id"] in set(rebuilt["image_id"])
    assert reasons[same.iloc[1]["image_id"]] == "duplicate_same_split"
    assert summary["nih_duplicates"]["removed_cross_split"] == 1
    assert (rebuilt["sha256"].fillna("") != "").all()

    table, _ = probes.origin_probe(dfs, data_root, out, n=10)
    assert list(table.columns) == probes.NORMALIZATIONS
    probes.normalization_preview(dfs, data_root, local, per_source=1)
    for f in (out / "20_tbx11k_class_stats.png", out / "21_22_hospital_fingerprint_by_normalization.png",
              local / "normalization_preview.png"):
        assert f.exists(), f
