"""Seeded synthetic weekly retail demand series.

The real Walmart series (see ``data_loader``) is the point of the project; this
module exists so the tests, the dashboard demo and the CI all work without
Kaggle credentials. Synthetic data is ALWAYS labelled as synthetic — it is
never presented as observed retail demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .holidays import WALMART_HOLIDAYS

# Sales uplift per Walmart event week (mirrors the real markdown spikes so the
# synthetic series exercises the same holiday structure as the real one).
_EVENT_UPLIFT: dict[str, float] = {
    "Super Bowl": 1.08,
    "Labor Day": 1.05,
    "Thanksgiving": 1.35,
    "Christmas": 1.45,
}


def normalize_weekly_ds(ds: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Snap dates back to the Friday of their week (the series' trading day)."""
    values = pd.Series(pd.to_datetime(ds))
    # dayofweek: Mon=0 .. Fri=4 .. Sun=6 -> subtract (dayofweek - 4) mod 7 days.
    offset = (values.dt.dayofweek - 4) % 7
    return values - pd.to_timedelta(offset, unit="D")


def generate_weekly_sales(
    n_weeks: int = 156,
    start: str = "2010-02-05",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a weekly demand series with known structure.

    The generative process is: growing local trend, a smooth 52-week seasonal
    cycle, four multiplicative event spikes, and multiplicative lognormal
    observation noise. It is deliberately close to the real Walmart series
    (trend + yearly seasonality + holiday spikes) while being fully reproducible
    and free of licensing constraints.

    Returns a DataFrame with columns ``ds`` (Friday timestamps), ``y`` (units)
    and ``IsHoliday`` (bool).
    """
    if n_weeks < 8:
        raise ValueError("n_weeks must be >= 8 to contain a seasonal cycle")

    rng = np.random.default_rng(seed)
    first = normalize_weekly_ds(pd.Series([pd.Timestamp(start)])).iloc[0]
    ds = pd.Series(pd.date_range(first, periods=n_weeks, freq="7D"))

    t = np.arange(n_weeks, dtype=float)
    level = 4_000_000.0
    trend = level * (1.0 + 0.0009 * t + 2.0e-6 * t**2)
    seasonal = (
        180_000.0 * np.cos(2 * np.pi * t / 52.0)
        + 70_000.0 * np.sin(4 * np.pi * t / 52.0)
    )
    signal = trend + seasonal

    factor = np.ones(n_weeks, dtype=float)
    holiday = np.zeros(n_weeks, dtype=bool)
    for name, uplift in _EVENT_UPLIFT.items():
        for date in WALMART_HOLIDAYS[name]:
            event = pd.Timestamp(date)
            hits = (ds == event).to_numpy()
            factor[hits] *= uplift
            holiday[hits] = True

    noise = rng.lognormal(mean=0.0, sigma=0.08, size=n_weeks)
    y = signal * factor * noise

    return pd.DataFrame({"ds": ds, "y": y, "IsHoliday": holiday})
