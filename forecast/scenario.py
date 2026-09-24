"""Scenario analysis on top of a forecast distribution.

"What if demand increases 20%?" is answered by scaling the *whole predictive
distribution* (point and both bands), not just the point line — a scenario
without a band is a guess, which is the thing this project exists to avoid.

Scenario outputs are counterfactual model-based estimates, never observations.
"""

from __future__ import annotations

import pandas as pd

from .schema import band_columns, level_tag

DEFAULT_LEVEL = 0.80


def apply_scenario(
    forecast_df: pd.DataFrame,
    factor: float = 1.2,
    dates=None,
) -> pd.DataFrame:
    """Scale a forecast table by ``factor`` (all weeks, or only ``dates``)."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    out = forecast_df.copy()
    mask = slice(None) if dates is None else out["ds"].isin(pd.to_datetime(list(dates)))
    numeric = [c for c in out.columns if c != "ds"]
    out.loc[mask, numeric] = out.loc[mask, numeric] * factor
    return out


def scenario_impact(
    base_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    level: float = DEFAULT_LEVEL,
) -> pd.DataFrame:
    """Per-week deltas between a scenario forecast and its baseline."""
    tag = level_tag(level)
    merged = base_df.merge(scenario_df, on="ds", suffixes=("_base", "_scen"))
    return pd.DataFrame(
        {
            "ds": merged["ds"],
            "yhat_base": merged["yhat_base"],
            "yhat_scenario": merged["yhat_scen"],
            "delta_yhat": merged["yhat_scen"] - merged["yhat_base"],
            f"delta_lower_{tag}": merged[f"lower_{tag}_scen"] - merged[f"lower_{tag}_base"],
            f"delta_upper_{tag}": merged[f"upper_{tag}_scen"] - merged[f"upper_{tag}_base"],
        }
    )
