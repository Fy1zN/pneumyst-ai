"""Occlusion Sensitivity: how much each image region supports a model's (calibrated) score.

A patch of the image is replaced by a baseline (a heavily blurred copy by default, because black squares look like
the padding the models were trained on), the image is scored again, and the score drop is spread back over the
patch. Overlapping patches are averaged. Positive values = the region raised the score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch

from ml.training.data import to_tensor

Calibrate = Callable[[np.ndarray], np.ndarray]  # (N, K) raw sigmoid scores -> (N, K) calibrated scores


@dataclass(frozen=True)
class OcclusionConfig:
    patch: int = 32
    stride: int = 16
    baseline: str = "blur"      # "blur" | "mean" | "black"
    blur_sigma: float = 12.0
    batch_size: int = 32
    max_oom_retries: int = 3     # halve the batch this many times on CUDA out-of-memory


@dataclass
class OcclusionResult:
    heatmaps: np.ndarray | None  # (K, H, W) float32 score drop, or None when invalid
    scores: np.ndarray           # (K,) calibrated scores of the unoccluded image
    status: str                  # "ok" or the reason the heatmap is not usable
    batch_size_used: int


def patch_positions(size: int, patch: int, stride: int) -> list[int]:
    """Top-left offsets along one axis; the last patch always reaches the image edge."""
    pos = list(range(0, max(size - patch, 0) + 1, stride))
    if pos[-1] + patch < size:
        pos.append(size - patch)
    return pos


def make_baseline(img: np.ndarray, cfg: OcclusionConfig) -> np.ndarray:
    if cfg.baseline == "blur":
        return cv2.GaussianBlur(img, (0, 0), cfg.blur_sigma)
    if cfg.baseline == "mean":
        return np.full_like(img, int(round(img.mean())))
    if cfg.baseline == "black":
        return np.zeros_like(img)
    raise ValueError(f"Unknown baseline: {cfg.baseline}")


def platt_calibrator(findings_cfg: dict, names: list[str]) -> Calibrate:
    """Calibrator from a *_calibration.json "findings" block, in the model's output order."""
    a = np.array([findings_cfg[n]["platt_a"] for n in names])
    b = np.array([findings_cfg[n]["platt_b"] for n in names])

    def calibrate(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))

    return calibrate


@torch.inference_mode()
def _scores(model, x: torch.Tensor, amp: bool) -> np.ndarray:
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp and x.device.type == "cuda"):
        return torch.sigmoid(model(x).float()).cpu().numpy()


def validate_heatmaps(heat: np.ndarray, shape: tuple[int, int]) -> str:
    if heat.ndim != 3 or heat.shape[1:] != shape:
        return "wrong_shape"
    if not np.isfinite(heat).all():
        return "non_finite"
    if np.all(heat.max(axis=(1, 2)) - heat.min(axis=(1, 2)) < 1e-6):
        return "constant"
    return "ok"


def occlusion_map(model, img: np.ndarray, device, calibrate: Calibrate | None = None,
                  cfg: OcclusionConfig = OcclusionConfig(), amp: bool = True) -> OcclusionResult:
    """Heatmaps for every model output of one uint8 HxW image."""
    model.eval()
    h, w = img.shape
    calibrate = calibrate or (lambda p: p)
    x0 = to_tensor(img).unsqueeze(0).to(device)
    xb = to_tensor(make_baseline(img, cfg)).unsqueeze(0).to(device)
    base_scores = calibrate(_scores(model, x0, amp))[0]

    ys, xs = patch_positions(h, cfg.patch, cfg.stride), patch_positions(w, cfg.patch, cfg.stride)
    boxes = [(y, x) for y in ys for x in xs]
    drops = np.zeros((len(boxes), len(base_scores)), dtype=np.float32)
    bs, retries, start = cfg.batch_size, 0, 0
    while start < len(boxes):
        chunk, batch = boxes[start:start + bs], None
        try:
            batch = x0.repeat(len(chunk), 1, 1, 1)
            for i, (y, x) in enumerate(chunk):
                batch[i, :, y:y + cfg.patch, x:x + cfg.patch] = xb[0, :, y:y + cfg.patch, x:x + cfg.patch]
            drops[start:start + len(chunk)] = base_scores - calibrate(_scores(model, batch, amp))
            start += len(chunk)
        except torch.cuda.OutOfMemoryError:
            batch = None
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            if retries >= cfg.max_oom_retries or bs == 1:
                return OcclusionResult(None, base_scores, "out_of_memory", bs)
            bs, retries = max(1, bs // 2), retries + 1

    heat = np.zeros((len(base_scores), h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for (y, x), d in zip(boxes, drops):
        heat[:, y:y + cfg.patch, x:x + cfg.patch] += d[:, None, None]
        count[y:y + cfg.patch, x:x + cfg.patch] += 1
    heat /= np.maximum(count, 1)
    status = validate_heatmaps(heat, (h, w))
    return OcclusionResult(heat if status == "ok" else None, base_scores, status, bs)


# ---------- checks ----------

def box_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """Boolean mask of an (x, y, w, h) box, clipped to the image."""
    x, y, bw, bh = box
    m = np.zeros(shape, dtype=bool)
    m[max(0, int(y)):min(shape[0], int(np.ceil(y + bh))), max(0, int(x)):min(shape[1], int(np.ceil(x + bw)))] = True
    return m


def pointing_hit(heat: np.ndarray, box) -> bool:
    """Is the hottest pixel inside the box?"""
    yx = np.unravel_index(np.argmax(heat), heat.shape)
    return bool(box_mask(heat.shape, box)[yx])


def energy_in_box(heat: np.ndarray, box) -> float:
    """Share of the positive heat that falls inside the box (chance level = box area share)."""
    pos = np.clip(heat, 0, None)
    total = pos.sum()
    return float(pos[box_mask(heat.shape, box)].sum() / total) if total > 0 else float("nan")


def smooth_random_map(shape: tuple[int, int], rng: np.random.Generator, sigma: float = 16.0) -> np.ndarray:
    """Blob-like random map with the same spatial scale as a heatmap (used for random deletion)."""
    return cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), sigma)


def deletion_score(model, img: np.ndarray, rank_map: np.ndarray, out_idx: int, device, frac: float = 0.10,
                   calibrate: Calibrate | None = None, cfg: OcclusionConfig = OcclusionConfig(),
                   amp: bool = True) -> float:
    """Calibrated score after replacing the top `frac` of pixels of rank_map with the baseline."""
    calibrate = calibrate or (lambda p: p)
    keep = rank_map < np.quantile(rank_map, 1 - frac)
    out = np.where(keep, img, make_baseline(img, cfg)).astype(np.uint8)
    return float(calibrate(_scores(model, to_tensor(out).unsqueeze(0).to(device), amp))[0, out_idx])


# ---------- display ----------

@dataclass(frozen=True)
class DisplayConfig:
    smooth_sigma: float = 8.0    # px; softens the 16 px patch grid so the map reads as regions, not a single peak
    full_red: float = 0.25       # colour scale floor: weaker regions than 25% of the score never reach full red
    min_peak: float = 0.05       # no map if no region removes at least 5% of the score (nothing strongly supports it)
    region_level: float = 0.5    # outline = area at least half as hot as the peak


@dataclass
class DisplayMap:
    heat: np.ndarray | None      # smoothed (H, W) score drop, or None when no map should be shown
    region: np.ndarray | None    # bool mask of the outlined region
    vmax: float                  # fixed colour scale for overlay()
    peak: float                  # strongest region's score drop as a share of the score
    reason: str                  # "shown" | "no_heatmap" | "no_band" | "below_possible" | "weak_evidence"


def display_map(heat: np.ndarray | None, score: float, possible_threshold: float | None,
                cfg: DisplayConfig = DisplayConfig()) -> DisplayMap:
    """Decide whether a finding's heatmap is worth showing, and prepare it for display.

    Only findings in the possible/likely band get a map: for low scores the map is mostly noise. The colour scale
    is the map's own peak, but never less than full_red * score, so weak evidence stays faint instead of turning red.
    """
    if heat is None:
        return DisplayMap(None, None, 0.0, 0.0, "no_heatmap")
    if possible_threshold is None:
        return DisplayMap(None, None, 0.0, 0.0, "no_band")
    if score < possible_threshold:
        return DisplayMap(None, None, 0.0, 0.0, "below_possible")
    smooth = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), cfg.smooth_sigma)
    top = float(smooth.max())
    peak = top / score if score > 0 else 0.0
    if peak < cfg.min_peak:
        return DisplayMap(None, None, 0.0, peak, "weak_evidence")
    return DisplayMap(smooth, smooth >= cfg.region_level * top, max(cfg.full_red * score, top), peak, "shown")


def overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45, vmax: float | None = None,
            region: np.ndarray | None = None) -> np.ndarray:
    """RGB uint8 image with positive heat drawn on top (red = raised the score most), plus an optional outline."""
    pos = np.clip(heat, 0, None)
    vmax = vmax or (pos.max() if pos.max() > 0 else 1.0)
    colour = cv2.applyColorMap((np.clip(pos / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    gray = np.repeat(img[..., None], 3, axis=2).astype(np.float32)
    a = alpha * np.clip(pos / vmax, 0, 1)[..., None]
    out = (gray * (1 - a) + colour * a).astype(np.uint8)
    if region is not None and region.any():
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out