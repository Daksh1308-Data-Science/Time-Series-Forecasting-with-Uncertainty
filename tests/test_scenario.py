import pandas as pd
import pytest

from forecast.benchmark import seasonal_naive_forecast
from forecast.data_generator import generate_weekly_sales
from forecast.scenario import apply_scenario, scenario_impact


def _base_forecast():
    train = generate_weekly_sales(n_weeks=120, seed=9)
    return seasonal_naive_forecast(train, horizon=6, period=52)


def test_apply_scenario_scales_point_and_bands():
    base = _base_forecast()
    scenario = apply_scenario(base, factor=1.2)
    for column in ["yhat", "lower_80", "upper_80", "lower_95", "upper_95"]:
        assert scenario[column].to_numpy() == pytest.approx(1.2 * base[column].to_numpy())
    assert scenario["ds"].equals(base["ds"])


def test_apply_scenario_can_target_dates():
    base = _base_forecast()
    target = base["ds"].iloc[:2]
    scenario = apply_scenario(base, factor=0.8, dates=target)
    assert scenario["yhat"].iloc[:2].to_numpy() == pytest.approx(0.8 * base["yhat"].iloc[:2])
    assert scenario["yhat"].iloc[2:].to_numpy() == pytest.approx(base["yhat"].iloc[2:])


def test_apply_scenario_rejects_non_positive_factor():
    with pytest.raises(ValueError):
        apply_scenario(_base_forecast(), factor=0)


def test_scenario_impact_reports_deltas():
    base = _base_forecast()
    scenario = apply_scenario(base, factor=1.2)
    impact = scenario_impact(base, scenario, level=0.80)
    assert list(impact.columns) == [
        "ds", "yhat_base", "yhat_scenario", "delta_yhat", "delta_lower_80", "delta_upper_80",
    ]
    assert impact["delta_yhat"].to_numpy() == pytest.approx(0.2 * base["yhat"].to_numpy())
    assert impact["delta_lower_80"].to_numpy() == pytest.approx(0.2 * base["lower_80"].to_numpy())
    assert impact["delta_upper_80"].to_numpy() == pytest.approx(0.2 * base["upper_80"].to_numpy())
