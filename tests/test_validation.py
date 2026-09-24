import numpy as np
import pandas as pd
import pytest

from forecast.data_generator import generate_weekly_sales
from forecast.validation import run_comparison, walk_forward_splits


def test_walk_forward_splits_are_chronological_and_disjoint():
    series = generate_weekly_sales(n_weeks=143, seed=42)
    folds = walk_forward_splits(series, horizon=6, min_train=104, step=6)
    assert len(folds) == 6
    for train, test in folds:
        assert len(train) == 104 or len(train) > 104
        assert len(test) == 6
        assert train["ds"].iloc[-1] < test["ds"].iloc[0]
    # expanding window: each fold trains on strictly more data
    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_walk_forward_splits_respects_max_folds():
    series = generate_weekly_sales(n_weeks=143, seed=42)
    folds = walk_forward_splits(series, horizon=6, min_train=104, step=6, max_folds=2)
    assert len(folds) == 2


def test_walk_forward_splits_rejects_insufficient_history():
    series = generate_weekly_sales(n_weeks=30, seed=42)
    with pytest.raises(ValueError):
        walk_forward_splits(series, horizon=6, min_train=104)


def test_run_comparison_closed_form_end_to_end():
    series = generate_weekly_sales(n_weeks=143, seed=42)
    result = run_comparison(
        series,
        models=("naive", "bayesian"),
        horizon=6,
        min_train=104,
        step=6,
        engine="closed_form",
        fourier_order=4,
    )
    table = result["table"]
    assert list(table["model"]) == ["bayesian", "naive"]  # sorted by MAE
    assert result["folds"] == 6
    assert result["ran"] == ["naive", "bayesian"]
    assert result["skipped"] == {}
    assert result["info"]["bayesian"]["engine"] == "closed_form"
    for column in ("mae", "rmse", "picp_80", "mpiw_80", "picp_95", "mpiw_95", "n_eval"):
        assert column in table.columns
        assert table[column].notna().all()
    # 95% bands must cover more actuals than 80% bands, for any sane model.
    assert (table["picp_95"] >= table["picp_80"]).all()
    # A wider nominal level cannot produce a narrower interval.
    assert (table["mpiw_95"] > table["mpiw_80"]).all()
    assert (table["n_eval"] == 36).all()


def test_run_comparison_coverage_is_plausible():
    series = generate_weekly_sales(n_weeks=143, seed=42)
    table = run_comparison(
        series,
        models=("bayesian",),
        horizon=6,
        min_train=104,
        step=6,
        engine="closed_form",
        fourier_order=4,
    )["table"]
    coverage = float(table["picp_95"].iloc[0])
    assert 0.6 <= coverage <= 1.0  # 36 out-of-sample points: wide tolerance by design
