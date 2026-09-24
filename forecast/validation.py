"""Walk-forward validation: measured coverage, width and accuracy per model.

Why walk-forward: a forecast interval can only be *calibrated* against data the
model has never seen, and a single holdout block is far too small to say
anything about a 95% coverage target. So we roll an expanding training window
forward, forecast a fixed horizon each time, pool the out-of-sample predictions
and compute PICP / MPIW / Winkler once over all folds. Every number in the
README comes from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bayesian_model import BayesianForecast
from .benchmark import seasonal_naive_forecast
from .metrics import interval_report, mae, rmse
from .schema import level_tag

DEFAULT_LEVELS = (0.80, 0.95)
MODEL_NAMES = ("naive", "bayesian", "prophet")

_MODEL_INFO = {
    "naive": "seasonal-naive point + empirical residual-quantile bands (frequentist)",
    "bayesian": "posterior predictive (parameter + observation uncertainty)",
    "prophet": "Prophet MAP fit + simulated trend variation",
}


def walk_forward_splits(
    df: pd.DataFrame,
    horizon: int = 6,
    min_train: int = 104,
    step: int = 6,
    max_folds: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds: (train, test) pairs, chronologically ordered."""
    n = len(df)
    if horizon < 1 or step < 1 or min_train < 2:
        raise ValueError("horizon/step >= 1 and min_train >= 2 required")
    if n < min_train + horizon:
        raise ValueError(
            f"need at least min_train + horizon = {min_train + horizon} rows, got {n}"
        )
    step = step or horizon
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    end = min_train
    while end + horizon <= n:
        train = df.iloc[:end].reset_index(drop=True)
        test = df.iloc[end : end + horizon].reset_index(drop=True)
        folds.append((train, test))
        if max_folds is not None and len(folds) >= max_folds:
            break
        end += step
    return folds


def _predict(
    model_name: str,
    train: pd.DataFrame,
    horizon: int,
    levels,
    period: int,
    engine: str,
    seed: int,
    holiday_dates,
    fourier_order: int,
    model_options: dict | None,
) -> tuple[pd.DataFrame, dict]:
    options = model_options or {}
    if model_name == "naive":
        return seasonal_naive_forecast(train, horizon=horizon, period=period, levels=levels), {}
    if model_name == "bayesian":
        model = BayesianForecast(
            train,
            period=period,
            order=fourier_order,
            holiday_dates=holiday_dates,
            engine=engine,
            seed=seed,
            **options,
        ).fit()
        return model.predict(periods=horizon, levels=levels), {"engine": model.engine}
    if model_name == "prophet":
        from .prophet_model import prophet_forecast  # local import: optional dep

        return (
            prophet_forecast(train, periods=horizon, levels=levels, **options),
            {},
        )
    raise ValueError(f"unknown model {model_name!r}; choose from {MODEL_NAMES}")


def run_comparison(
    df: pd.DataFrame,
    models=MODEL_NAMES,
    levels=DEFAULT_LEVELS,
    period: int = 52,
    horizon: int = 6,
    min_train: int = 104,
    step: int = 6,
    max_folds: int | None = 6,
    engine: str = "auto",
    seed: int = 42,
    holiday_dates=None,
    fourier_order: int = 6,
    model_options: dict | None = None,
) -> dict:
    """Pooled out-of-sample comparison table for the requested models.

    Returns ``{"table", "folds", "ran", "skipped", "info"}``. A model that
    cannot run (e.g. Prophet missing) is reported in ``skipped`` with the
    reason instead of silently dropping out of the table.
    """
    folds = walk_forward_splits(df, horizon, min_train, step, max_folds)
    collected: dict[str, list[pd.DataFrame]] = {m: [] for m in models}
    skipped: dict[str, str] = {}
    info: dict[str, dict] = {}

    for train, test in folds:
        for model_name in models:
            if model_name in skipped:
                continue
            try:
                pred, details = _predict(
                    model_name,
                    train,
                    horizon,
                    levels,
                    period,
                    engine,
                    seed,
                    holiday_dates,
                    fourier_order,
                    model_options.get(model_name) if model_options else None,
                )
            except Exception as exc:  # noqa: BLE001 - a failed model must not kill the run
                skipped[model_name] = f"{type(exc).__name__}: {exc}"
                continue
            merged = test.loc[:, ["ds", "y"]].merge(pred, on="ds", how="inner")
            if merged.empty:
                continue
            collected[model_name].append(merged)
            info.setdefault(model_name, {}).update(details)

    rows = []
    for model_name in models:
        if model_name in skipped or not collected[model_name]:
            continue
        data = pd.concat(collected[model_name], ignore_index=True)
        row = {
            "model": model_name,
            "mae": mae(data["y"], data["yhat"]),
            "rmse": rmse(data["y"], data["yhat"]),
            "n_eval": int(len(data)),
        }
        for level in levels:
            tag = level_tag(level)
            report = interval_report(
                data["y"], data[f"lower_{tag}"], data[f"upper_{tag}"], alpha=1.0 - level
            )
            row[f"picp_{tag}"] = report["picp"]
            row[f"mpiw_{tag}"] = report["mpiw"]
            row[f"winkler_{tag}"] = report["winkler"]
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("mae").reset_index(drop=True)
    return {
        "table": table,
        "folds": len(folds),
        "ran": [m for m in models if m not in skipped and collected[m]],
        "skipped": skipped,
        "info": info,
        "descriptions": {m: _MODEL_INFO[m] for m in models if m in _MODEL_INFO},
    }


def forecast_all(
    train_df: pd.DataFrame,
    periods: int = 6,
    models=MODEL_NAMES,
    levels=DEFAULT_LEVELS,
    period: int = 52,
    engine: str = "auto",
    seed: int = 42,
    holiday_dates=None,
    fourier_order: int = 6,
    model_options: dict | None = None,
) -> dict:
    """Full-history fits for the interactive dashboard: {model: forecast_df}."""
    forecasts: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for model_name in models:
        try:
            pred, _ = _predict(
                model_name,
                train_df,
                periods,
                levels,
                period,
                engine,
                seed,
                holiday_dates,
                fourier_order,
                model_options.get(model_name) if model_options else None,
            )
            forecasts[model_name] = pred
        except Exception as exc:  # noqa: BLE001
            skipped[model_name] = f"{type(exc).__name__}: {exc}"
    return {"forecasts": forecasts, "skipped": skipped}
