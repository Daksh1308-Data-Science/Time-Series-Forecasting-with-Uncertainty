"""Shared forecast-table conventions.

Every model in this project returns the same long table so the validation and
scenario layers can be model-agnostic:

    ds, yhat, lower_80, upper_80, lower_95, upper_95
"""

from __future__ import annotations

import pandas as pd

WEEK_DAYS = 7


def level_tag(level: float) -> str:
    """0.95 -> '95' (the column-name suffix for a central interval)."""
    return str(int(round(level * 100)))


def band_columns(levels) -> list[str]:
    cols: list[str] = []
    for level in levels:
        tag = level_tag(level)
        cols += [f"lower_{tag}", f"upper_{tag}"]
    return cols


def future_dates(last, periods: int) -> pd.Series:
    """The next ``periods`` weekly dates after ``last`` (7-day steps)."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    last = pd.Timestamp(last)
    dates = [last + pd.Timedelta(days=WEEK_DAYS * i) for i in range(1, periods + 1)]
    return pd.Series(pd.to_datetime(dates))


def standardize_forecast(df: pd.DataFrame, levels=(0.80, 0.95)) -> pd.DataFrame:
    """Reorder/validate a model forecast into the canonical column layout."""
    required = {"ds", "yhat", *band_columns(levels)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"forecast is missing columns: {sorted(missing)}")
    return df.loc[:, ["ds", "yhat", *band_columns(levels)]].reset_index(drop=True)
