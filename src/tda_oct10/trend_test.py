"""Trend tests for early-warning-signal indicators.

Implements the rolling Kendall-tau trend test used in Ismail et al. (2020)
to score the monotone increase of an EWS indicator on the run-up to a
critical transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DEFAULT_KENDALL_WINDOW = 125


def _scaled_window(window: int, scale_factor: float) -> int:
    """Apply ``scale_factor`` to a window length and validate the result."""
    effective = int(round(window * scale_factor))
    if effective < 2:
        raise ValueError(
            f"Effective window must be >= 2 (got {effective} from "
            f"window={window}, scale_factor={scale_factor})."
        )
    return effective


def kendall_tau_rolling(
    indicator_series: np.ndarray,
    window: int = DEFAULT_KENDALL_WINDOW,
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """Rolling Kendall's tau between indicator values and the time index.

    For each window of length ``round(window * scale_factor)``, the
    Kendall rank correlation between the indicator values inside the
    window and ``0, 1, ..., window-1`` is computed. A tau close to 1
    indicates a strong monotone increase of the indicator across that
    window -- the EWS pattern of interest.

    Windows containing any non-finite value yield ``NaN`` in both
    columns. The DataFrame index is the right-edge position of each
    window in the input series.

    Returns a DataFrame with columns ``['tau', 'p_value']``.
    """
    indicator_series = np.asarray(indicator_series, dtype=float).ravel()
    effective_window = _scaled_window(window, scale_factor)
    n = indicator_series.size
    if n < effective_window:
        return pd.DataFrame(columns=["tau", "p_value"])

    n_out = n - effective_window + 1
    tau = np.empty(n_out)
    pval = np.empty(n_out)
    time_idx = np.arange(effective_window, dtype=float)

    for i in range(n_out):
        w = indicator_series[i : i + effective_window]
        if not np.all(np.isfinite(w)):
            tau[i] = np.nan
            pval[i] = np.nan
            continue
        result = kendalltau(time_idx, w)
        tau[i] = result.correlation if hasattr(result, "correlation") else result[0]
        pval[i] = result.pvalue if hasattr(result, "pvalue") else result[1]

    right_edge = np.arange(effective_window - 1, n)
    return pd.DataFrame({"tau": tau, "p_value": pval}, index=right_edge)
