"""Early-warning-signal indicators following Ismail et al. (2020).

Implements three scalar indicators applied over a sliding window:

* lag-1 autocorrelation (AC1) -- canonical critical-slowing-down marker
* sample variance (VAR)       -- variance inflation near a transition
* mean power spectrum at low frequencies (MPS) -- spectral reddening

Defaults reproduce the Ismail et al. (2020) configuration: indicator window
N = 250, low-frequency band k = 2 .. floor(N/8) (k = 31 at N = 250).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_INDICATOR_WINDOW = 250
DEFAULT_MPS_K_MIN = 2
DEFAULT_MPS_K_MAX_FRACTION = 0.125  # k_max = floor(N/8)


def _scaled_window(window: int, scale_factor: float) -> int:
    """Apply ``scale_factor`` to a window length and validate the result."""
    effective = int(round(window * scale_factor))
    if effective < 2:
        raise ValueError(
            f"Effective window must be >= 2 (got {effective} from "
            f"window={window}, scale_factor={scale_factor})."
        )
    return effective


def autocorrelation_lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1D series.

    Returns NaN if the series has fewer than two points or zero variance.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return float("nan")

    a = x[:-1]
    b = x[1:]
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = np.sqrt((a_centered * a_centered).sum() * (b_centered * b_centered).sum())
    if denom == 0.0:
        return float("nan")
    return float((a_centered * b_centered).sum() / denom)


def variance(x: np.ndarray) -> float:
    """Sample variance (Bessel-corrected, ddof=1)."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return float("nan")
    return float(np.var(x, ddof=1))


def mean_power_spectrum_low_freq(
    x: np.ndarray,
    k_min: int = DEFAULT_MPS_K_MIN,
    k_max_fraction: float = DEFAULT_MPS_K_MAX_FRACTION,
) -> float:
    """Normalized power-spectrum mean over a low-frequency band.

    The series is mean-centered, then the periodogram |rFFT(x)|^2 is
    normalized so that the non-DC bins sum to one. The returned value is
    the arithmetic mean of bins ``k_min`` through ``floor(k_max_fraction * N)``
    inclusive. With the defaults (``k_min=2``, ``k_max_fraction=1/8``) and
    ``N = len(x) = 250`` this is the mean over ``k = 2 .. 31`` -- the band
    used by Ismail et al. (2020).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 4:
        return float("nan")

    centered = x - x.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2

    # Normalize across non-DC bins so MPS is independent of overall variance.
    non_dc_total = spectrum[1:].sum()
    if non_dc_total == 0.0:
        return float("nan")
    normalized = spectrum / non_dc_total

    k_max = int(np.floor(k_max_fraction * n))
    if k_max < k_min:
        return float("nan")
    k_max = min(k_max, normalized.size - 1)
    band = normalized[k_min : k_max + 1]
    if band.size == 0:
        return float("nan")
    return float(band.mean())


def rolling_indicators(
    series: np.ndarray,
    window: int = DEFAULT_INDICATOR_WINDOW,
    scale_factor: float = 1.0,
    k_min: int = DEFAULT_MPS_K_MIN,
    k_max_fraction: float = DEFAULT_MPS_K_MAX_FRACTION,
) -> pd.DataFrame:
    """Compute AC1, VAR, MPS over a sliding window of length ``window``.

    The effective window is ``round(window * scale_factor)``. Pass
    ``scale_factor=1.0`` to keep the default 250-step (e.g. daily) window;
    pass a larger factor to dilate the window for higher-frequency data
    (e.g. ``scale_factor=1440`` to span the same calendar time in minute
    samples).

    Returns a DataFrame with columns ``['AC1', 'VAR', 'MPS']`` and length
    ``len(series) - effective_window + 1``. The DataFrame index is the
    right-edge position of each window in the input series, so row ``i``
    summarizes ``series[i : i + effective_window]`` and is indexed by
    ``i + effective_window - 1``.
    """
    series = np.asarray(series, dtype=float).ravel()
    effective_window = _scaled_window(window, scale_factor)
    n = series.size
    if n < effective_window:
        return pd.DataFrame(columns=["AC1", "VAR", "MPS"])

    n_out = n - effective_window + 1
    ac1 = np.empty(n_out)
    var = np.empty(n_out)
    mps = np.empty(n_out)

    for i in range(n_out):
        w = series[i : i + effective_window]
        ac1[i] = autocorrelation_lag1(w)
        var[i] = variance(w)
        mps[i] = mean_power_spectrum_low_freq(w, k_min=k_min, k_max_fraction=k_max_fraction)

    right_edge = np.arange(effective_window - 1, n)
    return pd.DataFrame({"AC1": ac1, "VAR": var, "MPS": mps}, index=right_edge)
