"""Prophet wrapper: trend + seasonality + holiday events with built-in bands.

Prophet's uncertainty intervals come from its MAP fit plus simulated trend
variation (a frequentist-flavoured construction). We keep them as the
"practical baseline" the Bayesian posterior predictive is judged against.
"""

from __future__ import annotations

import pandas as pd

from .holidays import walmart_holidays
from .schema import band_columns, future_dates, level_tag

DEFAULT_LEVELS = (0.80, 0.95)


def prophet_forecast(
    train_df: pd.DataFrame,
    periods: int = 6,
    levels=DEFAULT_LEVELS,
    holidays: pd.DataFrame | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    """Fit Prophet per requested interval width and return the canonical table.

    Prophet samples one interval width per model, so we fit once per level
    (cheap) and stack the bands. Import errors propagate with an actionable
    message; ``validation.run_comparison`` catches them and reports the model
    as skipped.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "prophet is not installed. Install requirements-optional.txt; on "
            "Windows the first fit needs CmdStan: python -c \"import cmdstanpy; "
            "cmdstanpy.install_cxx_toolchain(force=True)\"."
        ) from exc

    frame = train_df.loc[:, ["ds", "y"]].copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["y"] = frame["y"].astype(float)
    holidays = walmart_holidays() if holidays is None else holidays

    future = pd.DataFrame({"ds": future_dates(frame["ds"].iloc[-1], periods)})
    options = dict(
        yearly_seasonality=True,
        weekly_seasonality=False,  # data is already weekly
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        uncertainty_samples=1000,
    )
    options.update(model_kwargs)

    levels = tuple(levels)
    out = pd.DataFrame({"ds": future["ds"]})
    for level in levels:
        model = Prophet(interval_width=level, **options)
        model.fit(frame)
        pred = model.predict(future)
        tag = level_tag(level)
        if "yhat" not in out:
            out["yhat"] = pred["yhat"].to_numpy(dtype=float)
        out[f"lower_{tag}"] = pred["yhat_lower"].to_numpy(dtype=float)
        out[f"upper_{tag}"] = pred["yhat_upper"].to_numpy(dtype=float)
    return out.loc[:, ["ds", "yhat", *band_columns(levels)]]
