"""Occlusion Sensitivity tests with a toy model whose answer depends on one known image region."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ml.explain.occlusion import (DisplayConfig, OcclusionConfig, deletion_score, display_map, energy_in_box,
                                  make_baseline, occlusion_map, overlay, patch_positions, platt_calibrator,
                                  pointing_hit, smooth_random_map, validate_heatmaps)


class RegionModel(nn.Module):
    """Two outputs: brightness of the top-left 32x32 corner, and of the bottom-right corner."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[:, 0, :32, :32].mean(dim=(1, 2)), x[:, 0, -32:, -32:].mean(dim=(1, 2))], dim=1) * 3


class OomModel(RegionModel):
    """Raises CUDA out-of-memory for batches larger than 4, like a too-small GPU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) > 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        return super().forward(x)


def _image(size: int = 96) -> np.ndarray:
    img = np.full((size, size), 60, np.uint8)
    img[:32, :32] = 240  # bright top-left corner: drives output 0
    return img


def test_patch_positions_reach_the_edge():
    assert patch_positions(320, 32, 16)[-1] == 288 and len(patch_positions(320, 32, 16)) == 19
    assert patch_positions(100, 32, 30) == [0, 30, 60, 68]


def test_heatmap_points_at_the_region_the_model_uses():
    res = occlusion_map(RegionModel(), _image(), "cpu", cfg=OcclusionConfig(patch=16, stride=8, batch_size=16))
    assert res.status == "ok" and res.heatmaps.shape == (2, 96, 96)
    heat = res.heatmaps[0]
    assert heat[:32, :32].mean() > 10 * abs(heat[48:, 48:]).mean()
    assert pointing_hit(heat, (0, 0, 32, 32))
    assert energy_in_box(heat, (0, 0, 32, 32)) > 0.8


def test_out_of_memory_halves_the_batch_and_still_finishes():
    res = occlusion_map(OomModel(), _image(), "cpu", cfg=OcclusionConfig(patch=16, stride=16, batch_size=16))
    assert res.status == "ok" and res.batch_size_used == 4


def test_out_of_memory_gives_up_after_retries():
    res = occlusion_map(OomModel(), _image(), "cpu", cfg=OcclusionConfig(patch=16, stride=16, batch_size=64,
                                                                         max_oom_retries=1))
    assert res.heatmaps is None and res.status == "out_of_memory" and np.isfinite(res.scores).all()


def test_validation_rejects_bad_maps():
    assert validate_heatmaps(np.zeros((2, 8, 8)), (8, 8)) == "constant"
    bad = np.random.default_rng(0).random((2, 8, 8))
    bad[0, 0, 0] = np.nan
    assert validate_heatmaps(bad, (8, 8)) == "non_finite"
    assert validate_heatmaps(np.random.default_rng(0).random((2, 8, 8)), (9, 8)) == "wrong_shape"


def test_deletion_of_top_region_beats_random():
    img, cfg = _image(), OcclusionConfig(patch=16, stride=8)
    res = occlusion_map(RegionModel(), img, "cpu", cfg=cfg)
    top = deletion_score(RegionModel(), img, res.heatmaps[0], 0, "cpu", frac=0.12, cfg=cfg)
    rand = deletion_score(RegionModel(), img, smooth_random_map(img.shape, np.random.default_rng(0)), 0, "cpu",
                          frac=0.12, cfg=cfg)
    assert top < rand < res.scores[0] + 1e-6


def test_calibrator_baselines_and_overlay():
    cal = platt_calibrator({"A": {"platt_a": 1.0, "platt_b": 0.0}, "B": {"platt_a": 2.0, "platt_b": -1.0}}, ["A", "B"])
    out = cal(np.array([[0.5, 0.5]]))
    assert out[0, 0] == pytest.approx(0.5) and out[0, 1] == pytest.approx(1 / (1 + np.e))
    img = _image()
    assert make_baseline(img, OcclusionConfig(baseline="black")).max() == 0
    assert make_baseline(img, OcclusionConfig(baseline="blur")).std() < img.std()
    rgb = overlay(img, np.random.default_rng(1).random((96, 96)))
    assert rgb.shape == (96, 96, 3) and rgb.dtype == np.uint8


def _region_heat(strength: float) -> np.ndarray:
    heat = np.random.default_rng(2).normal(0, 0.001, (96, 96)).astype(np.float32)
    heat[:32, :32] += strength
    return heat


def test_display_hides_low_bands_and_weak_maps():
    assert display_map(None, 0.5, 0.1).reason == "no_heatmap"
    assert display_map(_region_heat(0.2), 0.5, None).reason == "no_band"
    assert display_map(_region_heat(0.2), 0.05, 0.1).reason == "below_possible"
    weak = display_map(_region_heat(0.005), 0.5, 0.1)
    assert weak.reason == "weak_evidence" and weak.heat is None and weak.peak < DisplayConfig().min_peak


def test_display_outlines_the_supporting_region():
    dm = display_map(_region_heat(0.2), 0.5, 0.1)
    assert dm.reason == "shown" and dm.vmax == pytest.approx(dm.heat.max())  # strong map: no saturation
    assert dm.region[:24, :24].all() and not dm.region[48:, 48:].any()
    rgb = overlay(_image(), dm.heat, vmax=dm.vmax, region=dm.region)
    assert (rgb == 255).all(axis=2).any()  # white outline drawn


def test_fixed_scale_keeps_weak_heat_faint():
    weak = display_map(_region_heat(0.05), 0.5, 0.1)
    assert weak.reason == "shown" and weak.vmax == pytest.approx(0.125) and weak.vmax > weak.heat.max()
    img, heat = _image(), _region_heat(0.01)
    own_scale, fixed = overlay(img, heat), overlay(img, heat, vmax=0.2)
    assert np.abs(fixed.astype(int) - img[..., None]).mean() < np.abs(own_scale.astype(int) - img[..., None]).mean()