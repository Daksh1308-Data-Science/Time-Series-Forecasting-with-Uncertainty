"""Tests for the committed measured results (reports/) used by the deployed app."""

import pandas as pd
import pytest

from forecast.validation import (
    default_reports_dir,
    load_measured_table,
    measured_config_mismatches,
)


def test_load_measured_table_reads_the_committed_run():
    measured = load_measured_table()
    table, meta = measured["table"], measured["meta"]
    assert set(table["model"]) == {"prophet", "naive", "bayesian"}
    assert (table["n_eval"] > 0).all()
    assert meta["horizon"] == 6
    assert meta["folds"] == 6
    assert meta["n_eval"] == int(table["n_eval"].iloc[0])
    assert "Walmart" in (meta["source_label"] or "")
    # A 95% band cannot be narrower than its nested 80% band.
    assert (table["mpiw_95"] > table["mpiw_80"]).all()
    # And it must cover at least as much as the 80% band.
    assert (table["picp_95"] >= table["picp_80"]).all()


def test_load_measured_table_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="comparison_table.csv"):
        load_measured_table(tmp_path)


def test_load_measured_table_raises_on_missing_columns(tmp_path):
    pd.DataFrame({"model": ["naive"], "mae": [1.0]}).to_csv(
        tmp_path / "comparison_table.csv", index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        load_measured_table(tmp_path)


def test_load_measured_table_works_without_metrics_json(tmp_path):
    pd.DataFrame(
        {
            "model": ["naive"], "mae": [1.0], "rmse": [1.0], "n_eval": [6],
            "picp_80": [0.8], "mpiw_80": [2.0], "winkler_80": [3.0],
            "picp_95": [0.95], "mpiw_95": [4.0], "winkler_95": [5.0],
        }
    ).to_csv(tmp_path / "comparison_table.csv", index=False)
    meta = load_measured_table(tmp_path)["meta"]
    assert meta == {"n_eval": 6}  # no metrics.json -> no config claims to check


def test_mismatches_detect_changed_settings():
    meta = {
        "folds": 6, "horizon": 6, "min_train": 104,
        "engine_requested": "closed_form", "n_eval": 36,
    }
    assert measured_config_mismatches(
        meta, horizon=6, max_folds=6, engine="closed_form", min_train=104
    ) == []
    diffs = measured_config_mismatches(
        meta, horizon=12, max_folds=3, engine="pymc", min_train=52
    )
    assert len(diffs) == 4
    assert any("horizon" in d for d in diffs)
    assert any("folds" in d for d in diffs)
    assert any("engine" in d for d in diffs)
    assert any("min_train" in d for d in diffs)


def test_mismatches_tolerate_unknown_meta_values():
    """A table without metrics.json must not trigger false mismatch warnings."""
    assert measured_config_mismatches(
        {"n_eval": 36}, horizon=6, max_folds=4, engine="pymc", min_train=104
    ) == []


def test_default_reports_dir_contains_the_artifacts():
    reports = default_reports_dir()
    assert (reports / "comparison_table.csv").is_file()
    assert (reports / "metrics.json").is_file()
