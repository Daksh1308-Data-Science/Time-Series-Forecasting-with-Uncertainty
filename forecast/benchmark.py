"""Frequentist baselines: seasonal-naive point forecast + residual-based bands.

These are the "frequentist" reference point of the project. A seasonal-naive
forecast is the hard-to-beat sanity check on any retail series, and the
interval is a plain empirical quantile of the seasonal differences — no
sampling, no priors, just the observed error distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import future_dates, band_columns

DEFAULT_LEVELS = (0.80, 0.95)


def seasonal_naive_point(y: np.ndarray, horizon: int, period: int = 52) -> np.ndarray:
    """Repeat the last observed season: forecast week k = y[n - period + k - 1].

    Falls back to the last observation when the series is shorter than one
    season (so the function never reaches outside the data).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("training series is empty")
    if n <= period:
        return np.full(horizon, y[-1], dtype=float)
    base = y[n - period :]
    return np.array([base[(k - 1) % period] for k in range(1, horizon + 1)], dtype=float)


def seasonal_naive_forecast(
    train_df: pd.DataFrame,
    horizon: int = 6,
    period: int = 52,
    levels=DEFAULT_LEVELS,
) -> pd.DataFrame:
    """Seasonal-naive point forecast with empirical residual quantiles as bands."""
    y = train_df["y"].to_numpy(dtype=float)
    point = seasonal_naive_point(y, horizon, period)

    if y.size > period:
        resid = y[period:] - y[:-period]
    elif y.size > 1:
        resid = y[1:] - y[:-1]
    else:
        resid = np.zeros(1, dtype=float)

    dates = future_dates(pd.Timestamp(train_df["ds"].iloc[-1]), horizon)
    out = pd.DataFrame({"ds": dates, "yhat": point})
    for level in levels:
        alpha = 1.0 - level
        lo = np.quantile(resid, alpha / 2.0)
        hi = np.quantile(resid, 1.0 - alpha / 2.0)
        tag = int(round(level * 100))
        out[f"lower_{tag}"] = np.maximum(point + lo, 0.0)
        out[f"upper_{tag}"] = point + hi
    return out.loc[:, ["ds", "yhat", *band_columns(levels)]]
