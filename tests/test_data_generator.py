import numpy as np
import pandas as pd
import pytest

from forecast.data_generator import generate_weekly_sales, normalize_weekly_ds


def test_generate_is_deterministic_for_same_seed():
    a = generate_weekly_sales(n_weeks=60, seed=7)
    b = generate_weekly_sales(n_weeks=60, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generate_changes_with_seed():
    a = generate_weekly_sales(n_weeks=60, seed=1)
    b = generate_weekly_sales(n_weeks=60, seed=2)
    assert not np.allclose(a["y"].to_numpy(), b["y"].to_numpy())


def test_generate_shape_positive_and_finite():
    df = generate_weekly_sales(n_weeks=120, seed=42)
    assert list(df.columns) == ["ds", "y", "IsHoliday"]
    assert len(df) == 120
    assert (df["y"] > 0).all()
    assert np.isfinite(df["y"]).all()
    assert df["IsHoliday"].sum() > 0


def test_generate_rejects_tiny_series():
    with pytest.raises(ValueError):
        generate_weekly_sales(n_weeks=4)


def test_normalize_weekly_ds_snaps_to_friday():
    raw = pd.Series(pd.to_datetime(["2010-02-07", "2010-02-12", "2010-02-14"]))  # Sun, Fri, Sun
    snapped = normalize_weekly_ds(raw)
    assert list(snapped.dt.dayofweek) == [4, 4, 4]
    # Weeks are labelled by their Friday: a Sunday snaps back, never forward.
    assert snapped.iloc[0] == pd.Timestamp("2010-02-05")
    assert snapped.iloc[1] == pd.Timestamp("2010-02-12")
    assert snapped.iloc[2] == pd.Timestamp("2010-02-12")
