"""Screener tests with toy models: bands, input checks, partial failures, heatmap gating, JSON output."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from ml.datasets.targets import TB
from ml.inference.screener import InputLimits, LoadedModel, Screener, band, read_image
from ml.preprocessing.preprocess import load_gray, preprocess_array
from ml.training.model import build_model

CAL = {"platt_a": 1.0, "platt_b": 0.0, "possible_threshold": 0.3, "likely_threshold": 0.7, "status": "banded"}
NO_BAND = {**CAL, "status": "insufficient_data"}


class ConstModel(nn.Module):
    """Fixed probabilities for every image."""

    def __init__(self, probs: list[float]):
        super().__init__()
        p = torch.tensor(probs)
        self.register_buffer("logits", torch.log(p / (1 - p)))
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.logits.expand(len(x), -1)


class SpotModel(nn.Module):
    """One output: high when the small spot near the top-left of the image is bright."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :1, 24:72, 40:88].mean(dim=(1, 2, 3)) * 1.5 - 1).unsqueeze(1)


class BrokenModel(nn.Module):
    def forward(self, x):
        raise RuntimeError("boom")


class OomOnBatchModel(ConstModel):
    """Scores a single image fine but runs out of memory on occlusion batches."""

    def forward(self, x):
        if len(x) > 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return super().forward(x)


def _xray(size=(400, 360), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(120, 30, size).clip(0, 255).astype(np.uint8)
    img[40:80, 40:80] = 250  # bright spot near the top-left
    return img


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _screener(*models: LoadedModel) -> Screener:
    return Screener(list(models), device="cpu")


def test_band_edges():
    assert band(0.3, CAL) == "possible" and band(0.2999, CAL) == "unlikely"
    assert band(0.7, CAL) == "likely" and band(0.69, CAL) == "possible"
    assert band(0.99, NO_BAND) is None


def test_scores_bands_and_json():
    nih = LoadedModel("nih", ["Atelectasis", "Hernia"], {"Atelectasis": CAL, "Hernia": NO_BAND},
                      ConstModel([0.8, 0.5]))
    tb = LoadedModel("tb", [TB], {TB: CAL}, ConstModel([0.1]))
    res = _screener(nih, tb).screen(_png(_xray()), explain=False)
    assert res.status == "success" and res.stages["input"] == "success"
    assert res.finding("Atelectasis").score == pytest.approx(0.8, abs=1e-5)
    assert res.finding("Atelectasis").band == "likely" and res.finding("Atelectasis").group == "Lung tissue"
    assert res.finding("Hernia").band is None and res.finding("Hernia").band_status == "insufficient_data"
    assert res.finding(TB).band == "unlikely" and res.finding(TB).heatmap == "not_requested"
    out = json.loads(json.dumps(res.to_dict()))
    assert len(out["findings"]) == 3 and out["preprocess"]["size"] == 320 and "image" not in out

@pytest.mark.parametrize("data, reason", [
    (b"not an image", "unreadable"),
    (_png(_xray((100, 100))), "too_small"),
    (_png(np.full((400, 400), 90, np.uint8)), "blank_image"),
    (_png(_xray((300, 1000))), "bad_aspect_ratio"),
], ids=["unreadable", "too_small", "blank_image", "bad_aspect_ratio"])
def test_bad_inputs_are_rejected_before_any_model_runs(data, reason):
    model = ConstModel([0.5])
    res = _screener(LoadedModel("tb", [TB], {TB: CAL}, model)).screen(data)
    assert res.status == "failed" and res.error == reason and res.errors["input"] == "INVALID_IMAGE"
    assert model.calls == 0 and res.findings == []
    assert read_image(b"x" * 100, InputLimits(max_bytes=10)) == (None, "file_too_large")


def test_same_scores_from_path_bytes_and_array(tmp_path):
    arr = _xray()
    (tmp_path / "x.png").write_bytes(_png(arr))
    s = _screener(LoadedModel("tb", [TB], {TB: CAL}, SpotModel()))
    scores = [s.screen(src, explain=False).finding(TB).score for src in (tmp_path / "x.png", _png(arr), arr)]
    assert scores[0] == scores[1] == scores[2]


def test_preprocessing_matches_training_pipeline(tmp_path):
    arr16 = (_xray().astype(np.uint16) * 200)  # 16-bit PNG, like some hospital exports
    path = tmp_path / "x16.png"
    Image.fromarray(arr16).save(path)
    res = _screener(LoadedModel("tb", [TB], {TB: CAL}, ConstModel([0.5]))).screen(path, explain=False)
    expected, transform = preprocess_array(load_gray(path))
    assert np.array_equal(res.image, expected) and res.preprocess["transform"] == transform


def test_missing_or_failing_model_does_not_hide_the_other():
    ok = LoadedModel("tb", [TB], {TB: CAL}, ConstModel([0.9]))
    gone = LoadedModel("nih", ["Atelectasis"], {"Atelectasis": CAL}, None, "MODEL_UNAVAILABLE: FileNotFoundError")
    res = _screener(gone, ok).screen(_png(_xray()), explain=False)
    assert res.status == "partial" and res.stages["nih"] == "unavailable"
    a = res.finding("Atelectasis")
    assert a.status == "unavailable" and a.score is None and a.band is None and a.heatmap == "unavailable"
    assert res.finding(TB).band == "likely"

    broken = LoadedModel("nih", ["Atelectasis"], {"Atelectasis": CAL}, BrokenModel())
    res = _screener(broken, ok).screen(_png(_xray()), explain=False)
    assert res.stages["nih"] == "failed" and res.errors["nih"] == "INFERENCE_FAILED: RuntimeError"
    assert res.finding("Atelectasis").status == "failed" and res.finding(TB).score is not None

    res = _screener(broken).screen(_png(_xray()))
    assert res.status == "failed"


def test_load_marks_missing_checkpoint_unavailable(tmp_path):
    (tmp_path / "ml" / "configs").mkdir(parents=True)
    for run in ("a_run", "b_run"):
        (tmp_path / "ml" / "configs" / f"{run}.json").write_text(json.dumps(
            {"checkpoint": f"ml/checkpoints/{run}/best.pt", "findings": {TB: CAL}}))
    (tmp_path / "ml" / "checkpoints" / "a_run").mkdir(parents=True)
    torch.save({"model": build_model(1, pretrained=False).state_dict()}, tmp_path / "ml/checkpoints/a_run/best.pt")
    s = Screener.load(tmp_path, device="cpu", runs=[("a_run", "a_run.json", [TB]), ("b_run", "b_run.json", [TB])])
    assert s.models[0].model is not None and s.models[1].model is None
    assert s.models[1].error == "MODEL_UNAVAILABLE: FileNotFoundError"


def test_heatmap_only_for_possible_or_likely():
    spot = LoadedModel("tb", [TB], {TB: {**CAL, "possible_threshold": 0.5, "likely_threshold": 0.9}}, SpotModel())
    res = _screener(spot).screen(_png(_xray()))
    tb = res.finding(TB)
    assert tb.band in ("possible", "likely") and res.stages["explain:tb"] == "success"
    assert tb.heatmap == "shown" and tb.heatmap_peak > 0.05
    region = res.maps[TB].region
    assert region[32:64, 48:80].mean() > 0.5 and not region[160:, 160:].any()
    rgb = res.overlay(TB)
    assert rgb.shape == (320, 320, 3) and res.overlay("Atelectasis") is None

    quiet = ConstModel([0.05])
    res = _screener(LoadedModel("tb", [TB], {TB: CAL}, quiet)).screen(_png(_xray()))
    assert res.stages["explain:tb"] == "skipped" and res.finding(TB).heatmap == "below_possible"
    assert quiet.calls == 1 and res.maps == {} and res.status == "success"


def test_out_of_memory_keeps_scores():
    res = _screener(LoadedModel("tb", [TB], {TB: CAL}, OomOnBatchModel([0.9]))).screen(_png(_xray()))
    assert res.finding(TB).score == pytest.approx(0.9, abs=1e-5) and res.finding(TB).band == "likely"
    assert res.errors["explain:tb"] == "EXPLANATION_OOM" and res.finding(TB).heatmap == "unavailable"
    assert res.status == "partial"