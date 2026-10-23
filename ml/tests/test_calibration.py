"""Calibration tests on synthetic scores with a known miscalibration."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ml.evaluation.calibration import (apply_platt, expected_calibration_error, fit_platt, reliability_table,
                                       to_logit)


def _shifted_scores(n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """True probabilities, then pushed up and made overconfident (like pos_weight training does)."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-3, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(int)
    raw = 1 / (1 + np.exp(-(2.0 * true_logit + 3.0)))
    return y, raw


def test_platt_fixes_shift_and_overconfidence():
    y, raw = _shifted_scores()
    a, b = fit_platt(y, raw)
    assert 0.4 < a < 0.6 and -3.0 < b < -1.0  # roughly undoes logit * 2 + 3
    cal = apply_platt(raw, a, b)
    assert expected_calibration_error(y, cal) < 0.01 < expected_calibration_error(y, raw)


def test_platt_keeps_ranking():
    y, raw = _shifted_scores(2000)
    cal = apply_platt(raw, *fit_platt(y, raw))
    assert spearmanr(raw, cal).statistic > 0.999


def test_platt_ignores_unknown_and_handles_one_class():
    y = np.array([0, 1, np.nan, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.5, 0.2, 0.7, 0.3])
    a, b = fit_platt(y, p)
    assert np.isfinite(a) and np.isfinite(b)
    assert fit_platt(np.zeros(5), np.linspace(0.1, 0.5, 5)) == (1.0, 0.0)


def test_reliability_table_and_logit():
    t = reliability_table(np.array([0, 1, 1, 0]), np.array([0.05, 0.95, 0.9, 0.15]), n_bins=10)
    assert list(t["n"]) == [1, 1, 2] and t["observed_rate"].iloc[-1] == 1.0
    assert np.isfinite(to_logit([0.0, 1.0])).all()