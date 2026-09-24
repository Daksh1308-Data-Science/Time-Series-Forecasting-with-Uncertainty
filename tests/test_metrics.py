import numpy as np
import pytest

from forecast.metrics import interval_report, mae, mpiw, picp, rmse, winkler_score


def test_mae_known_value():
    # sklearn documented example: mean_absolute_error([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert mae([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_rmse_known_value():
    assert rmse([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert rmse([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_picp_counts_membership_only():
    # y=1 in [0,2] yes; y=2 in [0,3] yes; y=3 not in [10,11]; y=4 not in [0,1] -> 2/4
    assert picp([1, 2, 3, 4], [0, 0, 10, 0], [2, 3, 11, 1]) == pytest.approx(0.5)


def test_mpiw_known_value():
    assert mpiw([0, 0], [2, 4]) == pytest.approx(3.0)


def test_winkler_known_value_inside():
    # Gneiting & Raftery example: y=0.5, [0.3, 0.7], alpha=0.1 -> width only = 0.4
    assert winkler_score([0.5], [0.3], [0.7], alpha=0.1) == pytest.approx(0.4)


def test_winkler_known_value_outside():
    # y=0.8 above upper=0.7: 0.4 + (2/0.1)*(0.8-0.7) = 2.4
    assert winkler_score([0.8], [0.3], [0.7], alpha=0.1) == pytest.approx(2.4)


def test_winkler_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        winkler_score([1.0], [0.0], [2.0], alpha=1.0)


def test_interval_report_keys():
    report = interval_report([1, 2, 3], [0, 0, 0], [2, 2, 4], alpha=0.1)
    assert set(report) == {"picp", "mpiw", "winkler"}
    assert report["picp"] == pytest.approx(1.0)  # all three actuals are inside
    assert report["mpiw"] == pytest.approx(8 / 3)
