import numpy as np
import pandas as pd
import pytest

from forecast.decomposition import seasonal_strength, stl_decompose


def test_stl_decompose_shapes_and_alignment():
    y = pd.Series(np.arange(120, dtype=float))
    parts = stl_decompose(y, period=12, robust=False)
    for key in ("trend", "seasonal", "resid"):
        assert len(parts[key]) == len(y)
        assert parts[key].index.equals(y.index)
    reconstructed = parts["trend"] + parts["seasonal"] + parts["resid"]
    assert reconstructed.to_numpy() == pytest.approx(y.to_numpy(), rel=1e-6, abs=1e-6)


def test_stl_decompose_rejects_short_series():
    with pytest.raises(ValueError):
        stl_decompose(pd.Series(np.arange(10, dtype=float)), period=52)


def test_seasonal_strength_pure_sine_is_one():
    n = 240
    seasonal = pd.Series(3.0 * np.sin(2 * np.pi * np.arange(n) / 12))
    resid = pd.Series(np.zeros(n))
    assert seasonal_strength(seasonal, resid) == pytest.approx(1.0)


def test_seasonal_strength_pure_noise_is_zero():
    rng = np.random.default_rng(0)
    resid = pd.Series(rng.normal(size=240))
    seasonal = pd.Series(np.zeros(240))
    assert seasonal_strength(seasonal, resid) == pytest.approx(0.0)
