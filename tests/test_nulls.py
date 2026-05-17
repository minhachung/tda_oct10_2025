"""Correctness tests for the surrogate generators in ``tda_oct10.nulls``.

Verifies the four properties called out in the task brief for phase
randomization, plus marginal/spectrum guarantees for iAAFT, plus
block-bootstrap and empirical-p-value invariants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tda_oct10.nulls import (  # noqa: E402
    bootstrap_surrogates,
    compute_null_distribution_for_indicator,
    empirical_pvalue,
    iaaft_surrogates,
    phase_randomized_surrogates,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _ar1_series(T: int = 1024, phi: float = 0.7, seed: int = 0) -> np.ndarray:
    """AR(1) series with non-trivial autocorrelation."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.standard_normal()
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def _autocorr(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Biased *linear* ACF up to ``max_lag``, normalised so lag 0 = 1."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.dot(x, x) / x.size
    if var == 0.0:
        return np.zeros(max_lag + 1)
    acf = np.array(
        [np.dot(x[: x.size - k], x[k:]) / (x.size * var) for k in range(max_lag + 1)]
    )
    return acf


def _circular_autocorr(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Circular ACF via IFFT(|FFT|^2). Exactly preserved under phase
    randomization (Wiener-Khinchin theorem)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    spec = np.fft.rfft(x)
    full = np.fft.irfft(np.abs(spec) ** 2, n=x.size)
    if full[0] == 0.0:
        return np.zeros(max_lag + 1)
    return full[: max_lag + 1] / full[0]


# ---------------------------------------------------------------------------
# Phase randomization: the four properties from the brief
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [256, 257])  # even and odd lengths
def test_phase_randomized_surrogate_is_real_valued(T):
    """The irfft output must be real -- Hermitian symmetry enforced."""
    x = _ar1_series(T=T, seed=11)
    surr = phase_randomized_surrogates(x, n_surrogates=3, seed=0)
    assert np.isrealobj(surr)
    assert np.all(np.isfinite(surr))


@pytest.mark.parametrize("T", [256, 257])
def test_phase_randomized_power_spectrum_is_preserved(T):
    """Brief property 1: |FFT| of each surrogate matches the original."""
    x = _ar1_series(T=T, seed=12)
    target_amp = np.abs(np.fft.rfft(x))
    surr = phase_randomized_surrogates(x, n_surrogates=5, seed=0)
    for s in range(surr.shape[0]):
        amp = np.abs(np.fft.rfft(surr[s]))
        np.testing.assert_allclose(amp, target_amp, rtol=1e-10, atol=1e-10)


def test_phase_randomized_mean_and_variance_preserved():
    """Brief property 2: first two moments preserved.

    DC bin untouched -> exact mean preservation. Parseval ties total
    spectral power to variance, so variance is preserved to round-off.
    """
    x = _ar1_series(T=1024, seed=13)
    surr = phase_randomized_surrogates(x, n_surrogates=20, seed=0)
    for s in range(surr.shape[0]):
        assert surr[s].mean() == pytest.approx(x.mean(), abs=1e-10)
        # Bessel-corrected variance: matches to ~1e-12 since Parseval is exact.
        assert np.var(surr[s], ddof=1) == pytest.approx(np.var(x, ddof=1), rel=1e-10)


def test_phase_randomized_autocorrelation_preserved():
    """Brief property 3: ACF preserved (Wiener-Khinchin: ACF <-> |FFT|^2).

    The circular ACF (which is what Wiener-Khinchin actually equates to
    |FFT|^2) must be preserved exactly. The linear (biased) ACF, which
    is what people usually compute, will differ from the original by
    finite-sample edge effects but should still match closely.
    """
    x = _ar1_series(T=2048, seed=14)
    surr = phase_randomized_surrogates(x, n_surrogates=10, seed=0)

    circ_x = _circular_autocorr(x, max_lag=20)
    lin_x = _autocorr(x, max_lag=20)
    for s in range(surr.shape[0]):
        circ_s = _circular_autocorr(surr[s], max_lag=20)
        np.testing.assert_allclose(circ_s, circ_x, atol=1e-10)
        # Linear ACF: approximate (finite-sample edge effects) -- but the
        # shape is preserved to within ~2% on a strongly autocorrelated
        # T=2048 series.
        lin_s = _autocorr(surr[s], max_lag=20)
        np.testing.assert_allclose(lin_s, lin_x, atol=0.02)


def test_phase_randomized_marginal_may_differ():
    """Brief property 4: marginal distribution drifts toward Gaussian.

    A surrogate of a heavy-tailed (Laplace) series should typically have
    a smaller kurtosis than the original, which sanity-checks that we
    really are randomizing -- not just permuting.
    """
    rng = np.random.default_rng(15)
    x = rng.laplace(0, 1, size=2048)  # excess kurtosis ~ 3
    surr = phase_randomized_surrogates(x, n_surrogates=50, seed=0)
    orig_kurt = ((x - x.mean()) ** 4).mean() / (x.var() ** 2) - 3.0
    surr_kurts = np.array(
        [((s - s.mean()) ** 4).mean() / (s.var() ** 2) - 3.0 for s in surr]
    )
    # Mean surrogate kurtosis should be materially smaller (closer to 0)
    # than the Laplace original (excess kurt = 3).
    assert orig_kurt > 1.5
    assert surr_kurts.mean() < orig_kurt - 1.0


def test_phase_randomized_is_actually_random():
    """Each surrogate should differ from the original (and from each other)."""
    x = _ar1_series(T=512, seed=16)
    surr = phase_randomized_surrogates(x, n_surrogates=3, seed=0)
    for s in range(3):
        assert not np.allclose(surr[s], x)
    assert not np.allclose(surr[0], surr[1])


def test_phase_randomized_seed_is_deterministic():
    x = _ar1_series(T=256, seed=17)
    a = phase_randomized_surrogates(x, n_surrogates=4, seed=123)
    b = phase_randomized_surrogates(x, n_surrogates=4, seed=123)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Multivariate phase randomization (Prichard-Theiler / Schreiber-Schmitz 96)
# ---------------------------------------------------------------------------


def _two_correlated_series(T: int = 1024, seed: int = 21) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = _ar1_series(T=T, phi=0.7, seed=seed)
    noise = rng.standard_normal(T)
    y = 0.8 * x + 0.5 * noise
    return np.column_stack([x, y])


def test_multivariate_phase_randomization_preserves_cross_correlation():
    """Shared phases preserve cross-spectrum -> cross-correlation preserved."""
    data = _two_correlated_series(T=2048, seed=21)
    orig_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]

    surr = phase_randomized_surrogates(
        data, n_surrogates=20, seed=0, multivariate=True
    )
    surr_corrs = np.array([np.corrcoef(s[:, 0], s[:, 1])[0, 1] for s in surr])
    # Multivariate version preserves cross-correlation exactly (up to
    # the same round-off that preserves each marginal spectrum).
    np.testing.assert_allclose(surr_corrs, orig_corr, atol=1e-8)


def test_univariate_phase_randomization_destroys_cross_correlation():
    """Independent phases per column break the cross-correlation."""
    data = _two_correlated_series(T=2048, seed=22)
    orig_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    assert abs(orig_corr) > 0.5  # sanity: original *is* correlated

    surr = phase_randomized_surrogates(
        data, n_surrogates=50, seed=0, multivariate=False
    )
    surr_corrs = np.array([np.corrcoef(s[:, 0], s[:, 1])[0, 1] for s in surr])
    # Mean cross-corr should collapse to ~0 with independent phases.
    assert abs(surr_corrs.mean()) < 0.1


# ---------------------------------------------------------------------------
# iAAFT
# ---------------------------------------------------------------------------


def test_iaaft_preserves_marginal_exactly():
    rng = np.random.default_rng(31)
    x = rng.laplace(0, 1, size=512)
    surr = iaaft_surrogates(x, n_surrogates=5, seed=0, max_iter=200)
    for s in range(surr.shape[0]):
        np.testing.assert_allclose(np.sort(surr[s]), np.sort(x), atol=1e-12)


def test_iaaft_power_spectrum_is_close():
    x = _ar1_series(T=1024, phi=0.85, seed=32)
    target_amp = np.abs(np.fft.rfft(x))
    surr = iaaft_surrogates(x, n_surrogates=3, seed=0, max_iter=500, tol=1e-6)
    for s in range(surr.shape[0]):
        amp = np.abs(np.fft.rfft(surr[s]))
        # Relative L2 between magnitudes should be small.
        rel = np.linalg.norm(amp - target_amp) / np.linalg.norm(target_amp)
        assert rel < 0.05


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_shape_and_values_are_drawn_from_original():
    x = _ar1_series(T=200, seed=41)
    surr = bootstrap_surrogates(x, n_surrogates=4, seed=0, block_length=10)
    assert surr.shape == (4, 200)
    # Every value in the surrogate must come from the original series.
    original_set = set(x.tolist())
    for s in range(4):
        for v in surr[s]:
            assert v in original_set


def test_bootstrap_preserves_marginal_moments_approximately():
    x = _ar1_series(T=4096, seed=42)
    surr = bootstrap_surrogates(x, n_surrogates=20, seed=0)
    means = surr.mean(axis=1)
    stds = surr.std(axis=1, ddof=1)
    # Means cluster around the sample mean of x; stds around the sample std.
    assert abs(means.mean() - x.mean()) < 0.05
    assert abs(stds.mean() - x.std(ddof=1)) < 0.05


def test_bootstrap_2d_shape():
    data = _two_correlated_series(T=512, seed=43)
    surr = bootstrap_surrogates(data, n_surrogates=3, seed=0, block_length=8)
    assert surr.shape == (3, 512, 2)


# ---------------------------------------------------------------------------
# Empirical p-value
# ---------------------------------------------------------------------------


def test_pvalue_observed_above_all_null_is_minimum():
    null = np.linspace(-1.0, 1.0, 100)
    p = empirical_pvalue(observed=10.0, surrogate_distribution=null, alternative="greater")
    assert p == pytest.approx(1.0 / 101.0)


def test_pvalue_observed_below_all_null_is_one_for_greater():
    null = np.linspace(0.0, 1.0, 100)
    p = empirical_pvalue(observed=-5.0, surrogate_distribution=null, alternative="greater")
    # observed = -5.0 is below every surrogate -> all 100 surrogates >= observed.
    # p = (1 + 100) / (1 + 100) = 1.0.
    assert p == pytest.approx(1.0)


def test_pvalue_less_alternative():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    # observed = 1.5 -> 1 surrogate is <= 1.5 (the value 1.0).
    p = empirical_pvalue(observed=1.5, surrogate_distribution=null, alternative="less")
    assert p == pytest.approx((1 + 1) / (1 + 4))


def test_pvalue_two_sided_centered_observation_is_large():
    rng = np.random.default_rng(51)
    null = rng.standard_normal(1000)
    p_center = empirical_pvalue(observed=float(np.median(null)), surrogate_distribution=null, alternative="two-sided")
    p_tail = empirical_pvalue(observed=5.0, surrogate_distribution=null, alternative="two-sided")
    assert p_center > 0.5  # observed at the centre -> not significant
    assert p_tail < p_center


def test_pvalue_rejects_unknown_alternative():
    with pytest.raises(ValueError):
        empirical_pvalue(observed=0.0, surrogate_distribution=np.zeros(10), alternative="banana")


# ---------------------------------------------------------------------------
# Top-level driver (smoke test -- does not exercise TDAPipeline heavily)
# ---------------------------------------------------------------------------


def test_compute_null_distribution_runs_end_to_end():
    """Smoke-test the driver with a tiny TDA configuration."""
    rng = np.random.default_rng(61)
    returns = rng.standard_normal(200)

    def indicator(features):
        return float(np.nanmean(features["L1_H1"]))

    out = compute_null_distribution_for_indicator(
        returns,
        indicator_fn=indicator,
        surrogate_method="bootstrap",
        n_surrogates=4,
        seed=0,
        n_jobs=1,
        pipeline_kwargs=dict(
            window_size=30,
            max_edge_length=np.inf,
            homology_dims=[1],
            landscape_layers=3,
            landscape_bins=50,
            n_jobs=1,
            mode="takens",
            embedding_dim=3,
            time_delay=1,
        ),
        method_kwargs=dict(block_length=15),
    )
    assert out["method"] == "bootstrap"
    assert out["surrogates"].shape == (4, 200)
    assert out["null_distribution"].shape == (4,)
    assert np.all(np.isfinite(out["null_distribution"]))


def test_compute_null_distribution_rejects_unknown_method():
    with pytest.raises(ValueError):
        compute_null_distribution_for_indicator(
            np.zeros(50),
            indicator_fn=lambda d: 0.0,
            surrogate_method="not-a-method",
            n_surrogates=2,
        )
