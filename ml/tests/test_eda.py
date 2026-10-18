"""EDA smoke test on synthetic datasets."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ml.datasets.build_manifests import build_all
from ml.eda.run_eda import run_eda
from ml.tests.test_manifests import data_root  # noqa: F401  (pytest fixture)


def test_run_eda_writes_outputs(data_root: Path, tmp_path: Path):
    build_all(data_root, ["tbx11k", "shenzhen", "montgomery", "nih"], tbx_val=0.2)
    out, local = tmp_path / "figures", tmp_path / "local"
    summary = run_eda(data_root, out, local, sample_per_source=5)

    expected = ["01_split_sizes.png", "02_tb_balance.png", "03_nih_prevalence_by_split.png",
                "04_nih_cooccurrence.png", "05_nih_findings_per_image.png", "06_nih_ap_view_by_finding.png",
                "07_demographics.png", "08_nih_images_per_patient.png", "09_image_properties.png", "eda_summary.json"]
    for name in expected:
        assert (out / name).exists(), name
    for source in ("tbx11k", "shenzhen", "montgomery"):
        assert (local / f"samples_{source}.png").exists()
    assert not list(out.glob("samples_*")), "X-ray grids must not go to the repo folder"
    assert json.loads((out / "eda_summary.json").read_text())["nih_patients"]["patients"] > 0
    assert "test_to_train_ratio" in summary["nih_prevalence"]
