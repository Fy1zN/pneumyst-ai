"""Extra EDA check tests on synthetic datasets."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image

import ml.eda.checks as checks
from ml.datasets.build_manifests import build_all
from ml.eda.run_eda import load_manifests
from ml.tests.test_manifests import data_root  # noqa: F401  (pytest fixture)


def _plant_formats(root: Path) -> None:
    tb = root / "raw" / "tbx11k" / "TBX11K" / "imgs" / "tb" / "tb0002.png"
    Image.open(tb).convert("RGB").save(tb)
    sz = root / "raw" / "shenzhen" / "CXR_png" / "CHNCXR_0001_0.png"
    g = Image.open(sz).convert("L")
    inv = Image.fromarray(255 - np.asarray(g)).convert("P")
    inv.putpalette([v for i in range(256) for v in (255 - i,) * 3])
    inv.save(sz)


def test_extra_checks(data_root: Path, tmp_path: Path):
    build_all(data_root, ["tbx11k", "shenzhen", "montgomery", "nih"], tbx_val=0.2)
    _plant_formats(data_root)
    dfs = load_manifests(data_root / "processed" / "manifests")
    out, local = tmp_path / "out", tmp_path / "local"
    out.mkdir()
    local.mkdir()

    census = checks.mode_census(dfs, data_root)
    assert {"RGB", "P"} <= set(census["mode"])

    ch = checks.channel_check(census, data_root, out, n=10)
    assert ch.loc[ch["mode"] == "RGB", "is_gray"].all()

    inv = checks.inversion_check(census, data_root, out, local, n=50)
    p_rows = inv[inv["mode"] == "P"]
    assert p_rows["palette_inverted"].all()

    assert not checks.border_check(dfs, data_root, out, n=5).empty
    sizes = checks.bbox_sizes(data_root, out)
    assert "median_side_224" in sizes.columns

    checks.duplicate_review(dfs, data_root, local)
    nih = dfs["nih"]
    tables = checks.prevalence_by_group(nih[nih["split"] == "train"], out)
    assert set(tables) == {"age_band", "sex", "view"}
    assert "Any finding" in checks.followup_effect(nih, out).columns

    q, thresholds = checks.quality_stats(dfs, data_root, out, n=5)
    assert {"min_sharpness_p1", "mean_brightness_p1", "mean_brightness_p99"} == set(thresholds)

    mism = checks.reading_mismatches(dfs, data_root, local)
    assert "shenzhen:CHNCXR_0004_1.png" in set(mism["image_id"])

    for name in ["11_color_mode_check.png", "12_inversion_check.png", "13_borders_and_aspect.png",
                 "14_bbox_size_at_224.png", "16_prevalence_by_age_sex_view.png", "17_positivity_vs_followup.png",
                 "18_sharpness_exposure.png"]:
        assert (out / name).exists(), name
    assert (local / "reading_mismatches.png").exists()
    assert not list(out.glob("*mismatch*")) and not list(out.glob("duplicate_pairs*"))
