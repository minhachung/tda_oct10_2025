"""Unit tests for EWS indicators and the rolling Kendall-tau trend test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tda_oct10.indicators import (  # noqa: E402
    autocorrelation_lag1,
    mean_power_spectrum_low_freq,
    rolling_indicators,
    variance,
)
from tda_oct10.trend_test import kendall_tau_rolling  # noqa: E402


# ---------------------------------------------------------------------------
# Specification tests requested in the task brief.
# ---------------------------------------------------------------------------


def test_ac1_of_white_noise_is_approximately_zero():
    """Mean AC1 over many independent white-noise samples should be ~0."""
    rng = np.random.default_rng(20251010)
    n_realizations = 500
    series_length = 250
    estimates = np.fromiter(
        (
            autocorrelation_lag1(rng.standard_normal(series_length))
            for _ in range(n_realizations)
        ),
        dtype=float,
        count=n_realizations,
    )
    # Lag-1 AC of white noise is biased toward -1/N for finite samples,
    # but the average over many realizations is well within 0.02 of 0.
    assert abs(estimates.mean()) < 0.02


def test_variance_of_gaussian_sample_matches_sigma_squared():
    rng = np.random.default_rng(7)
    sigma = 2.5
    sample = rng.normal(loc=0.0, scale=sigma, size=10_000)
    estimated = variance(sample)
    assert estimated == pytest.approx(sigma**2, rel=0.05)


def test_kendall_tau_monotonically_increasing_series_is_one():
    series = np.arange(200, dtype=float)
    out = kendall_tau_rolling(series, window=125)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["tau", "p_value"]
    np.testing.assert_allclose(out["tau"].to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# Supporting correctness tests.
# ---------------------------------------------------------------------------


def test_ac1_matches_numpy_corrcoef_reference():
    rng = np.random.default_rng(0)
    # Construct an AR(1) series with known autocorrelation > 0.
    phi = 0.7
    n = 5_000
    eps = rng.standard_normal(n)
    series = np.empty(n)
    series[0] = eps[0]
    for i in range(1, n):
        series[i] = phi * series[i - 1] + eps[i]
    estimate = autocorrelation_lag1(series)
    reference = float(np.corrcoef(series[:-1], series[1:])[0, 1])
    assert estimate == pytest.approx(reference, rel=1e-12, abs=1e-12)
    # And the AR(1) coefficient is recovered.
    assert estimate == pytest.approx(phi, abs=0.05)


def test_mps_low_freq_band_for_default_window():
    """At N=250 the spectral band must run from k=2 to k=31 inclusive."""
    # Build a series whose energy is split between a low (k=10) and a high
    # (k=80) Fourier mode of identical amplitude. Low-frequency mass should
    # therefore be ~1/2 of the normalized non-DC spectrum, spread over
    # k = 2..31 (30 bins), giving a mean of ~0.5/30 = 0.01667.
    n = 250
    t = np.arange(n)
    low_k, high_k = 10, 80
    series = np.cos(2 * np.pi * low_k * t / n) + np.cos(2 * np.pi * high_k * t / n)
    mps = mean_power_spectrum_low_freq(series)
    assert mps == pytest.approx(0.5 / 30, rel=0.01)


def test_mps_k_max_floor_matches_ismail_2020():
    """floor(0.125 * 250) must equal 31 (Ismail 2020 spec)."""
    n = 250
    k_max = int(np.floor(0.125 * n))
    assert k_max == 31


def test_rolling_indicators_shape_and_columns():
    rng = np.random.default_rng(1)
    series = rng.standard_normal(1000)
    out = rolling_indicators(series, window=250)
    assert list(out.columns) == ["AC1", "VAR", "MPS"]
    assert len(out) == len(series) - 250 + 1
    assert np.all(np.isfinite(out.to_numpy()))


def test_rolling_indicators_returns_empty_when_series_too_short():
    out = rolling_indicators(np.arange(50, dtype=float), window=250)
    assert out.empty
    assert list(out.columns) == ["AC1", "VAR", "MPS"]


def test_rolling_indicators_scale_factor_dilates_window():
    rng = np.random.default_rng(2)
    series = rng.standard_normal(1000)
    base = rolling_indicators(series, window=100, scale_factor=1.0)
    dilated = rolling_indicators(series, window=100, scale_factor=2.0)
    direct = rolling_indicators(series, window=200, scale_factor=1.0)
    assert len(base) == 1000 - 100 + 1
    assert len(dilated) == 1000 - 200 + 1
    np.testing.assert_allclose(dilated.to_numpy(), direct.to_numpy())


def test_kendall_tau_rolling_shape_and_columns():
    rng = np.random.default_rng(3)
    series = rng.standard_normal(400)
    out = kendall_tau_rolling(series, window=125)
    assert list(out.columns) == ["tau", "p_value"]
    assert len(out) == len(series) - 125 + 1
    # tau is bounded; p-values lie in [0, 1].
    assert ((out["tau"] >= -1.0) & (out["tau"] <= 1.0)).all()
    assert ((out["p_value"] >= 0.0) & (out["p_value"] <= 1.0)).all()


def test_kendall_tau_rolling_strictly_decreasing_is_minus_one():
    series = np.arange(200, 0, -1, dtype=float)
    out = kendall_tau_rolling(series, window=125)
    np.testing.assert_allclose(out["tau"].to_numpy(), -1.0)


def test_kendall_tau_rolling_scale_factor():
    rng = np.random.default_rng(4)
    series = rng.standard_normal(400)
    direct = kendall_tau_rolling(series, window=200, scale_factor=1.0)
    scaled = kendall_tau_rolling(series, window=100, scale_factor=2.0)
    np.testing.assert_allclose(
        scaled["tau"].to_numpy(),
        direct["tau"].to_numpy(),
        equal_nan=True,
    )
