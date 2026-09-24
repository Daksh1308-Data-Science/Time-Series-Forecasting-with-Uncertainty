"""Forecast accuracy and interval-quality metrics.

Point accuracy: MAE / RMSE (scikit-learn).
Uncertainty quality: PICP (coverage), MPIW (sharpness) and the Winkler /
Gneiting interval score, which rewards coverage *and* punishes width.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in units of ``y``."""
    return float(mean_absolute_error(_as_array(y_true), _as_array(y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error — punishes large misses more than MAE."""
    return float(np.sqrt(mean_squared_error(_as_array(y_true), _as_array(y_pred))))


def picp(y_true, lower, upper) -> float:
    """Prediction Interval Coverage Probability: fraction of actuals in [lower, upper]."""
    y, lo, hi = _as_array(y_true), _as_array(lower), _as_array(upper)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lower, upper) -> float:
    """Mean Prediction Interval Width — the sharpness counterpart of PICP."""
    lo, hi = _as_array(lower), _as_array(upper)
    if lo.size == 0:
        return float("nan")
    return float(np.mean(hi - lo))


def winkler_score(y_true, lower, upper, alpha: float = 0.05) -> float:
    """Mean interval score for a central ``1 - alpha`` interval.

    ``IS = (u - l) + 2/alpha * (l - y)_+ + 2/alpha * (y - u)_+``

    Lower is better. This is the standard way to rank forecasts that must carry
    intervals: a trivially wide interval has perfect coverage but a bad score.
    """
    y, lo, hi = _as_array(y_true), _as_array(lower), _as_array(upper)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if y.size == 0:
        return float("nan")
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def interval_report(y_true, lower, upper, alpha: float = 0.05) -> dict:
    """All interval-quality numbers for one interval, as a flat dict."""
    return {
        "picp": picp(y_true, lower, upper),
        "mpiw": mpiw(lower, upper),
        "winkler": winkler_score(y_true, lower, upper, alpha=alpha),
    }
