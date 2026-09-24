import sys

import numpy as np
import pandas as pd
import pytest

from forecast.bayesian_model import BayesianForecast, build_design, resolve_engine
from forecast.data_generator import generate_weekly_sales


def test_resolve_engine_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_engine("stan")


def test_resolve_engine_auto_falls_back_without_pymc(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymc", None)  # `import pymc` now raises
    assert resolve_engine("auto") == "closed_form"
    assert resolve_engine("closed_form") == "closed_form"


def test_build_design_contains_expected_terms():
    t = np.arange(10, dtype=float)
    holidays = pd.DataFrame({"Thanksgiving": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
    X, names = build_design(t, denom=9, period=12, order=2, holidays=holidays)
    assert names == [
        "const", "trend", "cos1", "sin1", "cos2", "sin2", "holiday_Thanksgiving",
    ]
    assert X.shape == (10, 7)
    assert (X[:, 0] == 1).all()
    assert X[1, -1] == 1


def test_closed_form_predict_shape_and_band_ordering():
    train = generate_weekly_sales(n_weeks=120, seed=42)
    model = BayesianForecast(train, period=52, order=4, engine="closed_form", seed=0)
    model.fit()
    pred = model.predict(periods=6, levels=(0.80, 0.95), n_samples=600)
    assert list(pred.columns) == [
        "ds", "yhat", "lower_80", "upper_80", "lower_95", "upper_95",
    ]
    assert len(pred) == 6
    assert pred["ds"].iloc[0] > train["ds"].iloc[-1]
    for tag in ("80", "95"):
        assert (pred[f"lower_{tag}"] <= pred[f"yhat"]).all()
        assert (pred[f"yhat"] <= pred[f"upper_{tag}"]).all()
    # A 95% band must be wider than the nested 80% band.
    assert (pred["upper_95"] - pred["lower_95"] > pred["upper_80"] - pred["lower_80"]).all()


def test_predictive_samples_are_reproducible():
    train = generate_weekly_sales(n_weeks=100, seed=42)
    model = BayesianForecast(train, engine="closed_form", order=3, seed=11).fit()
    _, first = model.sample_predictive(periods=4, n_samples=300)
    _, second = model.sample_predictive(periods=4, n_samples=300)
    assert first.shape == (4, 300)
    assert np.allclose(first, second)


def test_holiday_terms_present_in_design():
    train = generate_weekly_sales(n_weeks=120, seed=42)
    model = BayesianForecast(train, engine="closed_form", order=2, seed=0).fit()
    assert "holiday_Thanksgiving" in model.names
    assert "holiday_Christmas" in model.names


def test_predictive_coverage_is_near_nominal():
    """The whole point of the project: out-of-sample 95% bands should cover ~95%."""
    series = generate_weekly_sales(n_weeks=260, seed=5)
    train, test = series.iloc[:208], series.iloc[208:]
    model = BayesianForecast(train, period=52, order=6, engine="closed_form", seed=1).fit()
    pred = model.predict(periods=52, levels=(0.95,), n_samples=4000)
    covered = np.mean(
        (test["y"].to_numpy() >= pred["lower_95"].to_numpy())
        & (test["y"].to_numpy() <= pred["upper_95"].to_numpy())
    )
    # Generous band: Monte-Carlo noise + the fact that one 52-week block is a
    # small sample for a coverage claim. If this fails, calibration is broken.
    assert 0.75 <= covered <= 1.0


def test_parameter_summary_reports_every_term():
    train = generate_weekly_sales(n_weeks=80, seed=42)
    model = BayesianForecast(train, engine="closed_form", order=2, seed=0).fit()
    summary = model.parameter_summary()
    assert list(summary.columns) == ["term", "mean", "sd", "lower_95", "upper_95"]
    assert len(summary) == len(model.names)
    assert (summary["lower_95"] <= summary["upper_95"]).all()
