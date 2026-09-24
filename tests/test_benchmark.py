import numpy as np
import pandas as pd
import pytest

from forecast.benchmark import seasonal_naive_forecast, seasonal_naive_point
from forecast.data_generator import generate_weekly_sales
from forecast.holidays import holiday_indicators, walmart_holidays


def test_seasonal_naive_repeats_last_season():
    y = np.arange(1, 105, dtype=float)  # two full 52-week seasons
    point = seasonal_naive_point(y, horizon=6, period=52)
    assert point == pytest.approx([53.0, 54.0, 55.0, 56.0, 57.0, 58.0])


def test_seasonal_naive_short_series_repeats_last_value():
    y = np.array([1.0, 2.0, 3.0])
    point = seasonal_naive_point(y, horizon=4, period=52)
    assert point == pytest.approx([3.0, 3.0, 3.0, 3.0])


def test_seasonal_naive_forecast_columns_and_ordering():
    train = generate_weekly_sales(n_weeks=120, seed=3)
    forecast = seasonal_naive_forecast(train, horizon=6, period=52)
    assert list(forecast.columns) == [
        "ds", "yhat", "lower_80", "upper_80", "lower_95", "upper_95",
    ]
    assert len(forecast) == 6
    assert forecast["ds"].is_monotonic_increasing
    assert (forecast["ds"].iloc[0] > train["ds"].iloc[-1])
    assert (forecast["lower_80"] <= forecast["upper_80"]).all()


def test_walmart_holidays_table_shape():
    table = walmart_holidays()
    assert set(table.columns) == {"holiday", "ds"}
    assert set(table["holiday"]) == {"Super Bowl", "Labor Day", "Thanksgiving", "Christmas"}


def test_holiday_indicators_flag_known_thanksgiving_week():
    dates = pd.Series(pd.to_datetime(["2010-11-26", "2010-12-03"]))
    flags = holiday_indicators(dates)
    assert flags.loc[0, "Thanksgiving"] == 1
    assert flags.loc[1, "Thanksgiving"] == 0
