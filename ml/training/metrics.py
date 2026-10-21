"""Validation metrics. Labels may contain NaN (unknown); those rows are ignored per label."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _known(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    keep = ~np.isnan(y)
    return y[keep].astype(int), p[keep]


def safe_auroc(y, p) -> float:
    y, p = _known(y, p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_auprc(y, p) -> float:
    y, p = _known(y, p)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def threshold_for_sensitivity(y, p, target: float = 0.90) -> float:
    """Highest threshold that still flags at least `target` of the positives (predict positive if p >= t)."""
    y, p = _known(y, p)
    pos = np.sort(p[y == 1])
    if not len(pos):
        return float("nan")
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def binary_report(y, p, threshold: float) -> dict:
    y, p = _known(y, p)
    pred = p >= threshold
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    sens, ppv = div(tp, tp + fn), div(tp, tp + fp)
    return {"n": int(len(y)), "positives": int(y.sum()), "threshold": round(float(threshold), 4),
            "auroc": safe_auroc(y, p), "auprc": safe_auprc(y, p),
            "sensitivity": sens, "specificity": div(tn, tn + fp), "ppv": ppv, "npv": div(tn, tn + fn),
            "f1": div(2 * ppv * sens, ppv + sens) if tp else 0.0, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def multilabel_auroc(y: np.ndarray, p: np.ndarray, names: list[str]) -> pd.Series:
    return pd.Series({n: safe_auroc(y[:, i], p[:, i]) for i, n in enumerate(names)}, name="auroc")


def banded_auroc(y, p, stat, n_bands: int = 3) -> pd.DataFrame:
    """AUROC inside quantile bands of an image statistic (sharpness, brightness).

    If AUROC collapses inside every band, the model was separating classes by that statistic.
    """
    bands = pd.qcut(pd.Series(np.asarray(stat, dtype=float)), n_bands, duplicates="drop")
    rows = []
    for band in bands.cat.categories:
        m = (bands == band).to_numpy()
        yk, _ = _known(np.asarray(y)[m], np.asarray(p)[m])
        rows.append({"band": str(band), "n": int(m.sum()), "positives": int(yk.sum()),
                     "auroc": safe_auroc(np.asarray(y)[m], np.asarray(p)[m])})
    return pd.DataFrame(rows)


def bootstrap_auroc(y, p, n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """(auroc, 2.5th, 97.5th percentile) over bootstrap resamples."""
    y, p = _known(y, p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return safe_auroc(y, p), float(lo), float(hi)