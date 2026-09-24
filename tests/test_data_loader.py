from pathlib import Path

import pandas as pd
import pytest

from forecast.data_loader import (
    aggregate_weekly_sales,
    download_walmart,
    kaggle_credentials_present,
    load_raw,
)


def test_credentials_check_returns_bool():
    assert isinstance(kaggle_credentials_present(), bool)


def test_load_raw_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)


def test_download_walmart_raises_without_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr("forecast.data_loader.kaggle_credentials_present", lambda: False)
    with pytest.raises(FileNotFoundError, match="kaggle.json"):
        download_walmart(raw_dir=tmp_path)


def test_aggregate_weekly_sales_sums_stores_and_carries_holidays():
    train = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Dept": [1, 1, 1],
            "Date": pd.to_datetime(["2012-11-23", "2012-11-23", "2012-11-30"]),
            "Weekly_Sales": [100.0, 50.0, 20.0],
            "IsHoliday": [True, True, False],
        }
    )
    features = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(["2012-11-23", "2012-11-30"]),
            "IsHoliday": [True, False],
        }
    )
    series = aggregate_weekly_sales(train, features)
    assert list(series.columns) == ["ds", "y", "IsHoliday"]
    assert series["y"].tolist() == [150.0, 20.0]
    assert series["IsHoliday"].tolist() == [True, False]
    # Both dates are Fridays in the source data and stay Fridays.
    assert set(series["ds"].dt.dayofweek) == {4}
