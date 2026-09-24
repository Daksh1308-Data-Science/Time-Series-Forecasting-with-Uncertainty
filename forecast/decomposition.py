"""Seasonal decomposition helpers (STL) for understanding the series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_decompose(y: pd.Series, period: int = 52, robust: bool = True) -> dict:
    """STL decomposition of a regular series.

    Returns a dict with ``trend``, ``seasonal``, ``resid`` (all aligned to
    ``y``'s index) and the fitted ``model``.
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    if len(y) < 2 * period:
        raise ValueError(
            f"need at least 2 * period ({2 * period}) observations, got {len(y)}"
        )
    model = STL(np.asarray(y, dtype=float), period=period, robust=robust).fit()
    index = y.index
    return {
        "trend": pd.Series(model.trend, index=index),
        "seasonal": pd.Series(model.seasonal, index=index),
        "resid": pd.Series(model.resid, index=index),
        "model": model,
    }


def seasonal_strength(seasonal: pd.Series, resid: pd.Series) -> float:
    """Strength of seasonality in [0, 1] (Hyndman & Athanasopoulos).

    ``1 - Var(resid) / Var(resid + seasonal)``, clipped to [0, 1]. A value near
    1 means the seasonal component explains almost all leftover variation.
    """
    resid = np.asarray(resid, dtype=float)
    seasonal = np.asarray(seasonal, dtype=float)
    denom = np.var(resid + seasonal)
    if denom == 0:
        return 0.0
    return float(np.clip(1.0 - np.var(resid) / denom, 0.0, 1.0))
