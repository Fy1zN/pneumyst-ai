"""One chest X-ray in, 15 calibrated findings out: scores, bands, statuses and (optionally) heatmaps.

    screener = Screener.load()           # once: both checkpoints + their frozen calibration configs
    result = screener.screen("cxr.png")  # per image; bad input or a failing model never raises
    result.to_dict()                     # JSON-safe, for the API and the audit record

The image goes through the same preprocess_array() as the training data, so scores match the evaluated models.
Each model and each heatmap reports its own status: one failure never hides or fakes another result.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ml.datasets.common import to_8bit_gray
from ml.datasets.targets import DISPLAY_NAMES, GROUPS, NIH_FINDINGS, TB
from ml.explain.occlusion import (DisplayConfig, DisplayMap, OcclusionConfig, display_map, occlusion_map, overlay,
                                  platt_calibrator)
from ml.preprocessing.preprocess import CONFIG as PREP_CONFIG
from ml.preprocessing.preprocess import VERSION as PREP_VERSION
from ml.preprocessing.preprocess import preprocess_array
from ml.training.data import to_tensor
from ml.training.model import build_model

REPO = Path(__file__).resolve().parents[2]
# (run name, calibration config, outputs in model order)
DEFAULT_RUNS = [("nih_densenet121_v1", "nih_v1_calibration.json", NIH_FINDINGS),
                ("tb_densenet121_v2", "tb_v2_calibration.json", [TB])]
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}
HEATMAP_ERRORS = {"out_of_memory": "EXPLANATION_OOM", "constant": "EXPLANATION_DEGENERATE",
                  "wrong_shape": "INVALID_HEATMAP", "non_finite": "INVALID_HEATMAP"}


@dataclass(frozen=True)
class InputLimits:
    max_bytes: int = 50 * 1024 * 1024
    min_side: int = 256
    max_side: int = 10000
    max_aspect: float = 3.0      # longer side / shorter side
    min_std: float = 5.0         # gray-level spread below this = blank image


@dataclass
class LoadedModel:
    name: str
    findings: list[str]
    calibration: dict            # the "findings" block of the calibration config
    model: torch.nn.Module | None
    error: str | None = None     # set when the model could not be loaded


@dataclass
class FindingResult:
    finding: str
    display_name: str
    group: str
    model: str
    status: str                  # "success" | "unavailable" | "failed"
    score: float | None          # calibrated
    band: str | None             # "unlikely" | "possible" | "likely"; None without a score or a reliable band
    band_status: str             # "banded" | "insufficient_data"
    possible_threshold: float | None
    likely_threshold: float | None
    heatmap: str                 # display_map reason, or "not_requested" | "unavailable"
    heatmap_peak: float | None = None


@dataclass
class ScreeningResult:
    status: str                  # "success" | "partial" | "failed"
    error: str | None = None     # INVALID_IMAGE reason when the input was rejected
    findings: list[FindingResult] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    preprocess: dict = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    image: np.ndarray | None = None                            # preprocessed 320 px image (not serialized)
    maps: dict[str, DisplayMap] = field(default_factory=dict)  # finding -> shown heatmap (not serialized)

    def finding(self, name: str) -> FindingResult:
        return next(f for f in self.findings if f.finding == name)

    def overlay(self, name: str) -> np.ndarray | None:
        """RGB heatmap overlay on the preprocessed image, or None when no map is shown for this finding."""
        dm = self.maps.get(name)
        return None if dm is None else overlay(self.image, dm.heat, vmax=dm.vmax, region=dm.region)

    def to_dict(self) -> dict:
        def clean(v):
            if isinstance(v, float):
                return round(v, 6)
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v

        return clean({"status": self.status, "error": self.error, "findings": [vars(f) for f in self.findings],
                      "stages": self.stages, "errors": self.errors, "models": self.models,
                      "preprocess": self.preprocess, "timings_ms": self.timings_ms})


def band(score: float, cfg: dict) -> str | None:
    """Result band from the frozen thresholds; None for findings without a reliable band."""
    if cfg.get("status") != "banded":
        return None
    if score >= cfg["likely_threshold"]:
        return "likely"
    return "possible" if score >= cfg["possible_threshold"] else "unlikely"


def read_image(source, limits: InputLimits = InputLimits()) -> tuple[np.ndarray | None, str | None]:
    """File path, bytes or array -> 8-bit gray array, or (None, reason) when the image is unusable."""
    try:
        if isinstance(source, np.ndarray):
            arr = np.asarray(to_8bit_gray(Image.fromarray(source)), dtype=np.uint8)
        else:
            raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
            if len(raw) > limits.max_bytes:
                return None, "file_too_large"
            with Image.open(io.BytesIO(raw)) as im:
                if max(im.size) > limits.max_side:
                    return None, "too_large"
                im.load()
                arr = np.asarray(to_8bit_gray(im), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, TypeError, Image.DecompressionBombError):
        return None, "unreadable"
    h, w = arr.shape[:2]
    if arr.ndim != 2 or min(h, w) < limits.min_side:
        return None, "too_small"
    if max(h, w) > limits.max_side:
        return None, "too_large"
    if max(h, w) / min(h, w) > limits.max_aspect:
        return None, "bad_aspect_ratio"
    if arr.std() < limits.min_std:
        return None, "blank_image"
    return arr, None


class Screener:
    def __init__(self, models: list[LoadedModel], device="cpu", amp: bool = True,
                 occlusion: OcclusionConfig = OcclusionConfig(), display: DisplayConfig = DisplayConfig(),
                 limits: InputLimits = InputLimits()):
        self.models, self.device, self.amp = models, torch.device(device), amp
        self.occlusion, self.display, self.limits = occlusion, display, limits
        self.calibrators = {m.name: platt_calibrator(m.calibration, m.findings) for m in models}

    @classmethod
    def load(cls, repo: Path = REPO, device=None, runs=DEFAULT_RUNS, **kwargs) -> "Screener":
        """Load every model it can; a missing or broken checkpoint makes only that model unavailable."""
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models = []
        for name, cal_file, findings in runs:
            cal = json.loads((Path(repo) / "ml" / "configs" / cal_file).read_text())
            ckpt_path = Path(repo) / cal["checkpoint"]
            try:
                model = build_model(len(findings), pretrained=False, memory_efficient=False)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"])
                models.append(LoadedModel(name, findings, cal["findings"], model.to(device).eval()))
            except (OSError, KeyError, RuntimeError) as exc:
                models.append(LoadedModel(name, findings, cal["findings"], None,
                                          f"MODEL_UNAVAILABLE: {type(exc).__name__}"))
        return cls(models, device, **kwargs)

    def _score(self, m: LoadedModel, x: torch.Tensor) -> np.ndarray:
        with torch.inference_mode(), torch.autocast(device_type=x.device.type, dtype=torch.float16,
                                                    enabled=self.amp and x.device.type == "cuda"):
            raw = torch.sigmoid(m.model(x).float()).cpu().numpy()
        return self.calibrators[m.name](raw)[0]

    def screen(self, source, explain: bool = True) -> ScreeningResult:
        t0 = time.perf_counter()
        res = ScreeningResult(status="success", models=[m.name for m in self.models])
        arr, reason = read_image(source, self.limits)
        if arr is None:
            res.status, res.error = "failed", reason
            res.stages["input"], res.errors["input"] = "failed", "INVALID_IMAGE"
            return res
        img, transform = preprocess_array(arr, PREP_CONFIG)
        res.image, res.stages["input"] = img, "success"
        res.preprocess = {"version": PREP_VERSION, "size": PREP_CONFIG.size, "transform": transform}
        res.timings_ms["preprocess"] = (time.perf_counter() - t0) * 1000
        x = to_tensor(img).unsqueeze(0).to(self.device)

        for m in self.models:
            t1 = time.perf_counter()
            scores, status = None, "success"
            if m.model is None:
                status, res.errors[m.name] = "unavailable", m.error or "MODEL_UNAVAILABLE"
            else:
                try:
                    scores = self._score(m, x)
                except Exception as exc:  # any model failure is reported, never turned into a negative result
                    status, res.errors[m.name] = "failed", f"INFERENCE_FAILED: {type(exc).__name__}"
            res.stages[m.name] = status
            res.timings_ms[m.name] = (time.perf_counter() - t1) * 1000
            rows = []
            for i, f in enumerate(m.findings):
                c = m.calibration[f]
                score = None if scores is None else float(scores[i])
                rows.append(FindingResult(
                    finding=f, display_name=DISPLAY_NAMES[f], group=GROUP_OF[f], model=m.name,
                    status="success" if scores is not None else ("unavailable" if m.model is None else "failed"),
                    score=score, band=None if score is None else band(score, c), band_status=c["status"],
                    possible_threshold=c.get("possible_threshold"), likely_threshold=c.get("likely_threshold"),
                    heatmap="unavailable" if score is None else "not_requested"))
            res.findings += rows
            if scores is not None and explain:
                self._explain(m, img, rows, res)

        failed = [s for s in res.stages.values() if s not in ("success", "skipped", "not_requested")]
        if all(f.score is None for f in res.findings):
            res.status = "failed"
        elif failed:
            res.status = "partial"
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return res

    def _explain(self, m: LoadedModel, img: np.ndarray, rows: list[FindingResult], res: ScreeningResult) -> None:
        """Occlusion only when at least one finding is possible or likely; the rest get a reason, not a map."""
        stage = f"explain:{m.name}"
        if not any(r.band in ("possible", "likely") for r in rows):
            for r in rows:
                r.heatmap = "below_possible" if r.band == "unlikely" else "no_band"
            res.stages[stage] = "skipped"
            return
        t1 = time.perf_counter()
        try:
            occ = occlusion_map(m.model, img, self.device, self.calibrators[m.name], self.occlusion, self.amp)
            status = occ.status
        except Exception as exc:  # a broken explanation must not lose the scores
            occ, status = None, f"failed: {type(exc).__name__}"
        res.timings_ms[stage] = (time.perf_counter() - t1) * 1000
        if occ is None or occ.heatmaps is None:
            res.stages[stage] = "failed"
            res.errors[stage] = HEATMAP_ERRORS.get(status, "EXPLANATION_FAILED")
            for r in rows:
                r.heatmap = "unavailable"
            return
        res.stages[stage] = "success"
        for i, r in enumerate(rows):
            thr = r.possible_threshold if r.band_status == "banded" else None
            dm = display_map(occ.heatmaps[i], r.score, thr, self.display)
            r.heatmap, r.heatmap_peak = dm.reason, (dm.peak if dm.reason in ("shown", "weak_evidence") else None)
            if dm.heat is not None:
                res.maps[r.finding] = dm