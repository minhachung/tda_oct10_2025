"""Null models for TDA-indicator significance testing.

Two surrogate families are implemented:

1. ``bootstrap_surrogates`` -- moving-block bootstrap (Kuensch 1989).
   Preserves the marginal distribution exactly but breaks long-range
   serial dependence. Reproduces the null used by Ismail et al. (2020).

2. ``phase_randomized_surrogates`` -- Fourier-phase randomization
   (Theiler et al. 1992) in univariate or multivariate (Prichard &
   Theiler 1994 / Schreiber & Schmitz 1996) flavours. Preserves the
   power spectrum exactly. The multivariate variant uses a *shared*
   random-phase vector across components, which additionally preserves
   the cross-spectrum (and therefore lagged cross-correlations).

A third helper, ``iaaft_surrogates``, implements the iterative
amplitude-adjusted Fourier-transform algorithm of Schreiber & Schmitz
(1996). It preserves both the marginal distribution and (to a
controllable tolerance) the power spectrum.

Significance is assessed via ``empirical_pvalue`` and the convenience
driver ``compute_null_distribution_for_indicator``, which farms the
TDA pipeline + indicator call across surrogates using joblib.

References
----------
Theiler, Eubank, Longtin, Galdrikian, Farmer (1992). "Testing for
nonlinearity in time series: the method of surrogate data." Physica D.

Prichard & Theiler (1994). "Generating surrogate data for time series
with several simultaneously measured variables." PRL 73, 951.

Schreiber & Schmitz (1996). "Improved surrogate data for nonlinearity
tests." PRL 77, 635.

Kuensch (1989). "The jackknife and the bootstrap for general
stationary observations." Annals of Statistics.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from joblib import Parallel, delayed

__all__ = [
    "bootstrap_surrogates",
    "phase_randomized_surrogates",
    "iaaft_surrogates",
    "empirical_pvalue",
    "compute_null_distribution_for_indicator",
]


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _as_2d(returns: np.ndarray) -> tuple[np.ndarray, bool]:
    """Coerce 1-D returns to ``(T, 1)``; return ``(arr, was_1d)``."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    if arr.ndim != 2:
        raise ValueError(
            f"returns must be 1-D or 2-D, got ndim={arr.ndim} (shape={arr.shape})."
        )
    return arr, False


def _maybe_squeeze(surrogates: np.ndarray, was_1d: bool) -> np.ndarray:
    """If the input was 1-D, drop the trailing singleton dimension."""
    if was_1d and surrogates.shape[-1] == 1:
        return surrogates[..., 0]
    return surrogates


# ---------------------------------------------------------------------------
# Moving-block bootstrap (Kuensch 1989)
# ---------------------------------------------------------------------------


def _default_block_length(T: int) -> int:
    """Politis & White (2004) rule of thumb: L ~ T^(1/3)."""
    return max(2, int(round(T ** (1.0 / 3.0))))


def bootstrap_surrogates(
    returns: np.ndarray,
    n_surrogates: int = 1000,
    seed: int = 42,
    block_length: Optional[int] = None,
) -> np.ndarray:
    """Moving-block bootstrap surrogates (Kuensch 1989).

    Draws ``ceil(T / L)`` blocks of length ``L`` uniformly at random
    (with replacement) from the original series, concatenates them, and
    truncates to ``T``. Preserves the marginal distribution exactly but
    destroys long-range autocorrelation.

    Parameters
    ----------
    returns : ndarray, shape ``(T,)`` or ``(T, d)``.
    n_surrogates : number of surrogates to draw.
    seed : RNG seed.
    block_length : block length ``L``. Defaults to ``round(T**(1/3))``.

    Returns
    -------
    ndarray of shape ``(n_surrogates, T, d)`` (or ``(n_surrogates, T)``
    if ``returns`` was 1-D).
    """
    arr, was_1d = _as_2d(returns)
    T, d = arr.shape
    if T < 2:
        raise ValueError(f"returns must have at least 2 time steps, got T={T}.")
    L = int(block_length) if block_length is not None else _default_block_length(T)
    if L < 1 or L > T:
        raise ValueError(f"block_length must satisfy 1 <= L <= T, got L={L}, T={T}.")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / L))
    max_start = T - L + 1

    out = np.empty((n_surrogates, T, d), dtype=arr.dtype)
    for s in range(n_surrogates):
        starts = rng.integers(0, max_start, size=n_blocks)
        # Build per-position index array of length n_blocks*L, then truncate.
        idx = (starts[:, None] + np.arange(L)[None, :]).reshape(-1)[:T]
        out[s] = arr[idx]
    return _maybe_squeeze(out, was_1d)


# ---------------------------------------------------------------------------
# Phase-randomized surrogates (Theiler 1992 / Prichard & Theiler 1994)
# ---------------------------------------------------------------------------


def _phase_randomize_column(
    x: np.ndarray,
    rng: np.random.Generator,
    shared_phases: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return one phase-randomized surrogate of a single 1-D series.

    Uses the real-FFT representation. The DC bin is left untouched so the
    mean of the surrogate equals the mean of the original. If ``T`` is
    even, the Nyquist bin is purely real and gets a random sign.
    ``shared_phases`` -- if provided, length ``T // 2 - 1`` (interior
    positive frequencies only) -- is used in place of fresh random
    phases. This is the multivariate Prichard-Theiler trick that
    preserves cross-spectra across columns.
    """
    T = x.shape[0]
    X = np.fft.rfft(x)
    n_bins = X.shape[0]  # T // 2 + 1

    # Interior bins: indices 1 .. n_bins - 2 (all of 1.. if T odd, since
    # then there is no separate Nyquist bin).
    if T % 2 == 0:
        interior_end = n_bins - 1  # Nyquist at n_bins - 1 is handled separately
    else:
        interior_end = n_bins  # no Nyquist; everything past DC is interior

    interior_slice = slice(1, interior_end)
    n_interior = interior_end - 1

    if shared_phases is None:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_interior)
    else:
        if shared_phases.shape != (n_interior,):
            raise ValueError(
                f"shared_phases must have length {n_interior}, "
                f"got {shared_phases.shape}."
            )
        phases = shared_phases

    # Multiplicative phase rotation (not replacement): X_new = X * exp(i*phi).
    # For univariate inputs this is equivalent to phase replacement modulo
    # 2*pi (the marginal of phi is uniform either way), but for the
    # multivariate Prichard-Theiler trick it is essential: rotating each
    # column by the *same* phi preserves X_k * conj(Y_k) (the cross-
    # spectrum), and therefore lagged cross-correlations.
    X_new = X.copy()
    X_new[interior_slice] = X[interior_slice] * np.exp(1j * phases)

    # Nyquist bin (T even): must remain real. The only phases consistent
    # with real output at Nyquist are 0 and pi -- equivalently, a sign
    # flip. We multiply by a +-1 sign so the magnitude (= |X[N/2]|) is
    # preserved and, for shared phases, the same sign is applied to every
    # column.
    if T % 2 == 0:
        nyq_idx = n_bins - 1
        if shared_phases is None:
            sign = float(rng.choice([-1.0, 1.0]))
        else:
            sign = 1.0 if np.cos(phases[-1]) >= 0.0 else -1.0
        X_new[nyq_idx] = X[nyq_idx] * sign

    return np.fft.irfft(X_new, n=T)


def phase_randomized_surrogates(
    returns: np.ndarray,
    n_surrogates: int = 1000,
    seed: int = 42,
    multivariate: bool = False,
) -> np.ndarray:
    """Phase-randomized (Theiler 1992) surrogates.

    Preserves the power spectrum of each column exactly. The marginal
    distribution will look approximately Gaussian (this is the linear-
    Gaussian null).

    Parameters
    ----------
    returns : ndarray, ``(T,)`` or ``(T, d)``.
    n_surrogates : number of surrogates.
    seed : RNG seed.
    multivariate : if True and ``d > 1``, use a *shared* random-phase
        vector across columns (Prichard & Theiler 1994). This preserves
        the cross-spectrum and therefore lagged cross-correlations. If
        False, each column is randomized independently.

    Returns
    -------
    ndarray of shape ``(n_surrogates, T, d)``  (or ``(n_surrogates, T)``
    if input was 1-D).
    """
    arr, was_1d = _as_2d(returns)
    T, d = arr.shape
    if T < 4:
        raise ValueError(f"phase randomization needs T >= 4, got T={T}.")
    rng = np.random.default_rng(seed)
    n_interior = (T // 2 - 1) if T % 2 == 0 else (T // 2)

    out = np.empty((n_surrogates, T, d), dtype=float)
    for s in range(n_surrogates):
        shared = None
        if multivariate and d > 1:
            shared = rng.uniform(0.0, 2.0 * np.pi, size=n_interior)
        for j in range(d):
            out[s, :, j] = _phase_randomize_column(
                arr[:, j], rng, shared_phases=shared
            )
    return _maybe_squeeze(out, was_1d)


# ---------------------------------------------------------------------------
# iAAFT (Schreiber & Schmitz 1996)
# ---------------------------------------------------------------------------


def _iaaft_column(
    x: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """One iAAFT surrogate of a single 1-D series.

    Alternates spectral and rank projections until the rank order stops
    changing or ``max_iter`` is reached. Output has exactly the same
    sorted values as ``x`` (marginal preservation), and a power spectrum
    close to ``|FFT(x)|`` up to ``tol`` in relative L2.
    """
    T = x.shape[0]
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))

    # Initial guess: a random permutation of the data.
    surrogate = rng.permutation(x)
    prev_ranks = np.argsort(np.argsort(surrogate))

    for _ in range(max_iter):
        # Spectral projection: replace magnitudes with target, keep phases.
        spec = np.fft.rfft(surrogate)
        spec_phase = np.angle(spec)
        spec_new = target_amp * np.exp(1j * spec_phase)
        candidate = np.fft.irfft(spec_new, n=T)

        # Rank projection: replace values by the sorted-original values
        # at the matching rank.
        ranks = np.argsort(np.argsort(candidate))
        surrogate = sorted_x[ranks]

        if np.array_equal(ranks, prev_ranks):
            # Spectrum-fit check: how close are we to the target?
            current_amp = np.abs(np.fft.rfft(surrogate))
            denom = np.linalg.norm(target_amp)
            if denom == 0.0:
                break
            rel = np.linalg.norm(current_amp - target_amp) / denom
            if rel < tol:
                break
        prev_ranks = ranks

    return surrogate


def iaaft_surrogates(
    returns: np.ndarray,
    n_surrogates: int = 1000,
    seed: int = 42,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Iterative AAFT surrogates (Schreiber & Schmitz 1996).

    Preserves the marginal distribution of each column *exactly* and the
    power spectrum to within ``tol`` (relative L2 on the rFFT
    magnitudes). Each column is treated independently -- a multivariate
    iAAFT is not implemented because there is no consensus formulation
    that preserves cross-spectra and marginals jointly.

    Parameters
    ----------
    returns : ndarray, ``(T,)`` or ``(T, d)``.
    n_surrogates : number of surrogates.
    seed : RNG seed.
    max_iter : iteration cap.
    tol : spectrum-match tolerance (relative L2 on magnitudes).

    Returns
    -------
    ndarray, ``(n_surrogates, T, d)`` (or ``(n_surrogates, T)`` if 1-D).
    """
    arr, was_1d = _as_2d(returns)
    T, d = arr.shape
    if T < 4:
        raise ValueError(f"iAAFT needs T >= 4, got T={T}.")

    rng = np.random.default_rng(seed)
    out = np.empty((n_surrogates, T, d), dtype=float)
    for s in range(n_surrogates):
        for j in range(d):
            out[s, :, j] = _iaaft_column(
                arr[:, j], rng, max_iter=max_iter, tol=tol
            )
    return _maybe_squeeze(out, was_1d)


# ---------------------------------------------------------------------------
# Empirical p-values
# ---------------------------------------------------------------------------


_ALTERNATIVES = ("greater", "less", "two-sided")


def empirical_pvalue(
    observed: float,
    surrogate_distribution: np.ndarray,
    alternative: str = "greater",
) -> float:
    """Empirical p-value of ``observed`` against a null sample.

    Uses the conservative ``(1 + #{surrogate >= observed}) / (1 + N)``
    convention (North et al. 2002) so the smallest reportable p-value
    is ``1 / (1 + N)`` instead of zero.

    Parameters
    ----------
    observed : the test statistic from the real data.
    surrogate_distribution : 1-D array of statistics under the null.
    alternative : ``'greater'`` (default), ``'less'`` or ``'two-sided'``.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}."
        )
    null = np.asarray(surrogate_distribution, dtype=float).ravel()
    null = null[~np.isnan(null)]
    n = null.size
    if n == 0:
        return float("nan")

    if alternative == "greater":
        k = int(np.sum(null >= observed))
    elif alternative == "less":
        k = int(np.sum(null <= observed))
    else:  # two-sided: compare absolute deviations from the null median
        center = float(np.median(null))
        k = int(np.sum(np.abs(null - center) >= abs(observed - center)))
    return (1.0 + k) / (1.0 + n)


# ---------------------------------------------------------------------------
# Top-level driver: null distribution of an indicator
# ---------------------------------------------------------------------------


_SURROGATE_METHODS = {
    "bootstrap": bootstrap_surrogates,
    "phase_random": phase_randomized_surrogates,
    "iaaft": iaaft_surrogates,
}


def _run_one_surrogate(
    surrogate: np.ndarray,
    indicator_fn: Callable[..., Any],
    pipeline_kwargs: dict,
) -> Any:
    """Run the TDA pipeline + indicator on one surrogate.

    Imports the pipeline lazily so the joblib worker does not need to
    pickle the (heavy) giotto-tda transformer with the parent process.
    """
    from .tda_pipeline import TDAPipeline  # local import for joblib workers

    pipeline = TDAPipeline(**pipeline_kwargs)
    features = pipeline.fit_transform(surrogate)
    return indicator_fn(features)


def compute_null_distribution_for_indicator(
    returns: np.ndarray,
    indicator_fn: Callable[..., Any],
    surrogate_method: str,
    n_surrogates: int = 1000,
    pipeline_kwargs: Optional[dict] = None,
    seed: int = 42,
    n_jobs: int = -1,
    method_kwargs: Optional[dict] = None,
) -> dict:
    """Generate surrogates, run the TDA pipeline on each, collect indicators.

    Parameters
    ----------
    returns : ndarray, ``(T,)`` or ``(T, d)``.
    indicator_fn : callable. Receives the dict returned by
        ``TDAPipeline.fit_transform`` and returns a scalar (or any
        picklable value).
    surrogate_method : ``'bootstrap'``, ``'phase_random'`` or ``'iaaft'``.
    n_surrogates : number of draws.
    pipeline_kwargs : kwargs forwarded to ``TDAPipeline``. Defaults to ``{}``.
    seed : RNG seed used to draw the surrogates.
    n_jobs : passed to ``joblib.Parallel``. ``-1`` uses all cores.
    method_kwargs : extra kwargs forwarded to the surrogate generator
        (e.g. ``{'block_length': 25}`` for bootstrap, or
        ``{'multivariate': True}`` for phase randomization).

    Returns
    -------
    dict with keys ``surrogates`` (shape ``(n_surrogates, T, d)`` or
    ``(n_surrogates, T)``), ``null_distribution`` (1-D array of indicator
    values, length ``n_surrogates``), and ``method``.
    """
    if surrogate_method not in _SURROGATE_METHODS:
        raise ValueError(
            f"surrogate_method must be one of {tuple(_SURROGATE_METHODS)}, "
            f"got {surrogate_method!r}."
        )
    generator = _SURROGATE_METHODS[surrogate_method]
    method_kwargs = dict(method_kwargs or {})
    pipeline_kwargs = dict(pipeline_kwargs or {})

    surrogates = generator(
        returns, n_surrogates=n_surrogates, seed=seed, **method_kwargs
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_surrogate)(surrogates[s], indicator_fn, pipeline_kwargs)
        for s in range(n_surrogates)
    )
    null = np.asarray(results, dtype=float)

    return {
        "surrogates": surrogates,
        "null_distribution": null,
        "method": surrogate_method,
    }
