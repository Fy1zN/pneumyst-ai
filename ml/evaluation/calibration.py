"""Score calibration for sigmoid outputs (Platt scaling) and calibration checks.

Training with pos_weight pushes scores up for rare findings, so a raw 0.7 does not mean 70%.
Platt scaling fits calibrated = sigmoid(a * logit(raw) + b) on validation data: the slope fixes
over- or under-confidence and the intercept fixes the pos_weight shift. Ranking (AUROC) is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _known(y, p) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    keep = ~np.isnan(y)
    return y[keep].astype(int), p[keep]


def to_logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y, p) -> tuple[float, float]:
    """(a, b) so that sigmoid(a * logit(p) + b) is calibrated on (y, p)."""
    y, p = _known(y, p)
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(to_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * to_logit(p) + b)))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted score vs observed positive rate in equal-width score bins."""
    y, p = _known(y, p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = [{"bin": b, "n": int((bins == b).sum()), "mean_score": float(p[bins == b].mean()),
             "observed_rate": float(y[bins == b].mean())} for b in range(n_bins) if (bins == b).any()]
    return pd.DataFrame(rows, columns=["bin", "n", "mean_score", "observed_rate"])


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Average gap between score and observed rate, weighted by bin size (0 = perfect)."""
    t = reliability_table(y, p, n_bins)
    return float((t["n"] * (t["mean_score"] - t["observed_rate"]).abs()).sum() / max(t["n"].sum(), 1))