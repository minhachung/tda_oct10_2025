"""Phase-randomized + bootstrap surrogate test of the Oct 10 L^1 H_1 precursor.

Session 9 — null-distribution companion to Session 8.

Session 8 found a slow, 4 – 6 hour pre-cascade build-up in
:math:`\\|\\lambda\\|_1^{H_1}` at minute resolution (peak 16.95x baseline at
2025-10-10 17:14 UTC, hourly means 3 – 7x baseline through the
15:15 – 21:15 UTC run-up). The open question is whether this elevation
is statistically distinguishable from noise, or whether a comparable
4 – 6 hour stretch above 7x baseline can be produced by chance in calm
data.

This module answers the question with two surrogate families:

1. **Multivariate phase randomization** (Prichard & Theiler 1994 trick on
   top of Theiler 1992): preserves each column's power spectrum *and*
   the cross-spectrum between symbols. This is the more demanding null
   for a multivariate TDA pipeline, because lagged cross-correlations
   between BTC/ETH/SOL/BNB/XRP/DOGE are the structure the point cloud
   actually sees.
2. **Moving-block bootstrap** (Kuensch 1989; Ismail et al. 2020 style):
   preserves the marginal distribution exactly but destroys long-range
   serial dependence. A less strict baseline.

The iAAFT surrogate (``tda_oct10.nulls.iaaft_surrogates``) is *not* used
here because there is no consensus multivariate iAAFT formulation; an
independent-column iAAFT would destroy the cross-correlations that drive
the multivariate Vietoris-Rips topology. Phase randomization with the
Prichard-Theiler shared-phase trick is the correct multivariate
generalisation of "marginal-and-spectrum-preserving" surrogates for this
pipeline.

Test statistic
--------------
:math:`T_{\\mathrm{run\\_max}}` = the longest consecutive stretch (in
fractional hours, = minutes / 60) over which the 60-minute rolling mean
of :math:`\\|\\lambda\\|_1^{H_1}` stays strictly above ``7 * baseline``,
where the *baseline* is the median of :math:`\\|\\lambda\\|_1^{H_1}` over
the first 3 hours (= 180 indicator windows) of the series in question.
For surrogates, this is the median of the surrogate's own L^1 H_1 head;
for the observed precursor, it is the Session 8 baseline value (median
over 2025-10-08 00:00 – 03:00 UTC).

Observed window
---------------
The observed statistic is computed on the actual 2025-10-10
``PRECURSOR_START`` .. ``PRECURSOR_END`` window (17:14 – 21:14 UTC). The
cascade minute itself (21:15 UTC) is excluded so the observed run length
reflects pre-cascade dynamics only.

Empirical p-value
-----------------
:math:`p = (1 + \\#\\{s : T_s \\geq T_\\mathrm{obs}\\}) / (1 + N)` --
the conservative North et al. (2002) convention used elsewhere in this
project (see ``empirical_pvalue`` in ``tda_oct10.nulls``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tda_oct10.analysis_oct10_minute import (
    BASELINE_END,
    BASELINE_START,
    CASCADE_CENTER,
    HOMOLOGY_DIMS,
    INDICATOR_WINDOW,
    MAX_EDGE_LENGTH,
    SYMBOLS_OCT10,
    WINDOW_SIZE,
)
from tda_oct10.nulls import (
    bootstrap_surrogates,
    empirical_pvalue,
    phase_randomized_surrogates,
)

__all__ = [
    "CALM_START",
    "CALM_END",
    "PRECURSOR_START",
    "PRECURSOR_END",
    "RUN_MULTIPLIER",
    "SURROGATE_BASELINE_HEAD_MINUTES",
    "DEFAULT_N_SURROGATES",
    "DEFAULT_N_DISPLAY",
    "SurrogateRunResult",
    "Session9Result",
    "run_session9_surrogate_analysis",
    "plot_surrogate_test",
    "write_surrogate_findings_report",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALM_START = pd.Timestamp("2025-10-08 00:00", tz="UTC")
# Session 8 returns frame starts at 00:01 (first diff), so the calm window
# in *returns* space starts at 00:01 and ends at 16:59 UTC on Oct 10 (i.e.
# strictly less than 17:00 UTC, the precursor onset). Use 17:00 UTC as a
# half-open upper bound when slicing the returns DataFrame.
CALM_END = pd.Timestamp("2025-10-10 17:00", tz="UTC")

# Observed precursor window: 17:14 UTC (Session 8 peak L^1 H_1) through
# one minute before the cascade. Inclusive at both ends.
PRECURSOR_START = pd.Timestamp("2025-10-10 17:14", tz="UTC")
PRECURSOR_END = CASCADE_CENTER - pd.Timedelta(minutes=1)

# T_run_max threshold: rolling mean > 7 * baseline.
RUN_MULTIPLIER = 7.0
# Baseline window in *L^1 H_1 series* indices, used for surrogates. 180
# minutes = 3 hours, matching the Session 8 BASELINE_HOURS=3 convention.
SURROGATE_BASELINE_HEAD_MINUTES = 180

DEFAULT_N_SURROGATES = 1000
DEFAULT_N_DISPLAY = 50  # surrogate L^1 H_1 traces overlaid on observed in panel 3
DEFAULT_SEED = 42

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETURNS_PATH = _REPO_ROOT / "data" / "processed" / "oct10_minute_returns.parquet"
DEFAULT_FIGURE_PATH = _REPO_ROOT / "paper" / "figures" / "oct10_surrogate_test.png"
DEFAULT_REPORT_PATH = _REPO_ROOT / "paper" / "oct10_surrogate_findings.md"


# ---------------------------------------------------------------------------
# T_run_max statistic
# ---------------------------------------------------------------------------


def _longest_true_run(mask: np.ndarray) -> int:
    """Length (in samples) of the longest run of consecutive ``True`` in ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int((ends - starts).max())


def _rolling_mean(series: np.ndarray, window: int = INDICATOR_WINDOW) -> np.ndarray:
    """``window``-minute rolling mean with right-edge alignment; warmed-up only."""
    return pd.Series(series).rolling(window, min_periods=window).mean().to_numpy()


def _surrogate_baseline(l1_h1: np.ndarray) -> float:
    """Median of ``l1_h1`` over the first ``SURROGATE_BASELINE_HEAD_MINUTES`` samples."""
    head = l1_h1[:SURROGATE_BASELINE_HEAD_MINUTES]
    head = head[np.isfinite(head)]
    if head.size == 0:
        return float("nan")
    return float(np.median(head))


def _t_run_max(l1_h1: np.ndarray, threshold: float) -> float:
    """Fractional-hours version of the consecutive-above-threshold statistic."""
    if not np.isfinite(threshold) or threshold <= 0.0:
        return float("nan")
    rolling = _rolling_mean(l1_h1, window=INDICATOR_WINDOW)
    above = (rolling > threshold) & np.isfinite(rolling)
    return _longest_true_run(above) / 60.0


# ---------------------------------------------------------------------------
# Surrogate driver
# ---------------------------------------------------------------------------


def _run_one_surrogate_pipeline(surrogate: np.ndarray, pipeline_kwargs: dict) -> np.ndarray:
    """Run the TDA pipeline on one surrogate and return its L^1 H_1 series.

    Imports ``TDAPipeline`` lazily so each joblib worker constructs its own
    giotto-tda transformer chain without pickling state from the parent.
    """
    from tda_oct10.tda_pipeline import TDAPipeline  # noqa: WPS433 (lazy)

    pipeline = TDAPipeline(**pipeline_kwargs)
    features = pipeline.fit_transform(surrogate)
    return np.asarray(features["L1_H1"], dtype=float)


def _pipeline_kwargs_for_surrogates() -> dict:
    """Match the Session 8 pipeline configuration; ``n_jobs=1`` to avoid oversubscription."""
    return dict(
        mode="multivariate",
        window_size=WINDOW_SIZE,
        max_edge_length=MAX_EDGE_LENGTH,
        homology_dims=HOMOLOGY_DIMS,
        n_jobs=1,
    )


def _generate_surrogates(
    returns_2d: np.ndarray, *, method: str, n: int, seed: int
) -> np.ndarray:
    if method == "phase_random":
        return phase_randomized_surrogates(
            returns_2d, n_surrogates=n, seed=seed, multivariate=True
        )
    if method == "bootstrap":
        return bootstrap_surrogates(returns_2d, n_surrogates=n, seed=seed)
    raise ValueError(f"unknown surrogate method: {method!r}")


def _compute_null_distribution(
    surrogates: np.ndarray, *, n_jobs: int = -1
) -> np.ndarray:
    """Run the pipeline on each surrogate; return ``(n, n_windows)`` L^1 H_1 array."""
    pipeline_kwargs = _pipeline_kwargs_for_surrogates()
    n = surrogates.shape[0]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_surrogate_pipeline)(surrogates[s], pipeline_kwargs)
        for s in range(n)
    )
    return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# Observed L^1 H_1 series
# ---------------------------------------------------------------------------


def _real_l1_h1_series(returns: pd.DataFrame) -> pd.Series:
    """Run the Session 8 pipeline on the full 5-day minute frame, return L^1 H_1."""
    from tda_oct10.tda_pipeline import TDAPipeline  # noqa: WPS433 (lazy)

    pipeline = TDAPipeline(
        mode="multivariate",
        window_size=WINDOW_SIZE,
        max_edge_length=MAX_EDGE_LENGTH,
        homology_dims=HOMOLOGY_DIMS,
        n_jobs=-1,
    )
    features = pipeline.fit_transform(returns.to_numpy())
    l1_h1 = features["L1_H1"]
    n_windows = l1_h1.shape[0]
    window_ends = returns.index[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + n_windows]
    return pd.Series(l1_h1, index=window_ends, name="L1_H1")


def _baseline_median_from_real_series(series: pd.Series) -> float:
    mask = (series.index >= BASELINE_START) & (series.index <= BASELINE_END)
    sub = series.loc[mask].dropna()
    return float(sub.median()) if not sub.empty else float("nan")


def _observed_t_run_max(
    real_l1_h1: pd.Series, *, baseline: float, multiplier: float = RUN_MULTIPLIER
) -> tuple[float, int]:
    """Observed T_run_max on the precursor window; also return the rolling-mean count."""
    rolling = real_l1_h1.rolling(INDICATOR_WINDOW, min_periods=INDICATOR_WINDOW).mean()
    precursor = rolling.loc[
        (rolling.index >= PRECURSOR_START) & (rolling.index <= PRECURSOR_END)
    ]
    threshold = multiplier * baseline
    above = (precursor > threshold) & precursor.notna()
    arr = above.to_numpy()
    longest = _longest_true_run(arr)
    return longest / 60.0, int(arr.sum())


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SurrogateRunResult:
    """Per-method bundle of surrogate L^1 H_1 series and the empirical p-value.

    Attributes
    ----------
    method :
        ``'phase_random'`` or ``'bootstrap'``.
    l1_h1_array :
        ``(n_surrogates, n_windows)`` array of per-surrogate L^1 H_1 series.
    baselines :
        ``(n_surrogates,)`` per-surrogate baselines (median over first
        ``SURROGATE_BASELINE_HEAD_MINUTES`` of L^1 H_1).
    t_run_max :
        ``(n_surrogates,)`` per-surrogate T_run_max statistics (in hours).
    observed_t_run_max :
        Scalar T_run_max on the actual Oct 10 precursor window.
    p_value :
        Conservative empirical p-value
        ``(1 + #{T_s >= T_obs}) / (1 + n_surrogates)``.
    """

    method: str
    l1_h1_array: np.ndarray
    baselines: np.ndarray
    t_run_max: np.ndarray
    observed_t_run_max: float
    p_value: float


@dataclass
class Session9Result:
    """Bundle of both surrogate runs plus shared observed-side metadata."""

    phase: SurrogateRunResult
    bootstrap: SurrogateRunResult
    observed_baseline: float
    observed_t_run_max: float
    observed_precursor_above_count: int
    observed_l1_h1: pd.Series
    n_surrogates: int
    figure_path: Optional[Path] = None
    report_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _per_surrogate_stats(l1_h1_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = l1_h1_array.shape[0]
    baselines = np.empty(n, dtype=float)
    t_run = np.empty(n, dtype=float)
    for s in range(n):
        l1_h1 = l1_h1_array[s]
        b = _surrogate_baseline(l1_h1)
        baselines[s] = b
        t_run[s] = _t_run_max(l1_h1, multiplier_threshold(b))
    return baselines, t_run


def multiplier_threshold(baseline: float, multiplier: float = RUN_MULTIPLIER) -> float:
    """Convenience: ``multiplier * baseline``, used for thresholding."""
    return multiplier * baseline


def _run_method(
    returns_2d: np.ndarray,
    *,
    method: str,
    n_surrogates: int,
    seed: int,
    observed_t_run_max: float,
    n_jobs: int,
) -> SurrogateRunResult:
    logger.info("generating %d %s surrogates", n_surrogates, method)
    surrogates = _generate_surrogates(
        returns_2d, method=method, n=n_surrogates, seed=seed
    )
    logger.info("running TDA pipeline on %d surrogates (n_jobs=%d)", n_surrogates, n_jobs)
    l1_h1_array = _compute_null_distribution(surrogates, n_jobs=n_jobs)
    baselines, t_run_arr = _per_surrogate_stats(l1_h1_array)
    pval = empirical_pvalue(
        observed_t_run_max, t_run_arr, alternative="greater"
    )
    return SurrogateRunResult(
        method=method,
        l1_h1_array=l1_h1_array,
        baselines=baselines,
        t_run_max=t_run_arr,
        observed_t_run_max=observed_t_run_max,
        p_value=pval,
    )


def run_session9_surrogate_analysis(
    log_returns: Optional[pd.DataFrame] = None,
    *,
    returns_path: Optional[Path] = DEFAULT_RETURNS_PATH,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
    figure_path: Optional[Path] = DEFAULT_FIGURE_PATH,
    report_path: Optional[Path] = DEFAULT_REPORT_PATH,
    n_display: int = DEFAULT_N_DISPLAY,
) -> Session9Result:
    """End-to-end Session 9 surrogate test.

    Loads (or accepts) the 5-day minute log-returns frame, splits off the
    calm period [2025-10-08 00:00, 2025-10-10 17:00) UTC, runs
    ``n_surrogates`` phase-randomized and bootstrap surrogates through the
    Session 8 TDA pipeline in parallel, computes a per-surrogate
    :math:`T_{\\mathrm{run\\_max}}` statistic, and reports empirical
    p-values against the observed precursor signal.

    Parameters
    ----------
    log_returns :
        Optional minute-resolution log-returns DataFrame. If ``None``,
        loads from ``returns_path``.
    returns_path :
        Path to the cached parquet file (Session 8 output).
    n_surrogates :
        Number of surrogates per method. Defaults to 1000.
    seed :
        RNG seed for surrogate generation.
    n_jobs :
        ``joblib.Parallel`` ``n_jobs``. ``-1`` uses all cores.
    figure_path, report_path :
        Where to write the 3-panel figure and the findings markdown.
        Pass ``None`` to skip.
    n_display :
        Number of surrogate L^1 H_1 traces to overlay in panel 3.
    """
    if log_returns is None:
        if returns_path is None or not Path(returns_path).exists():
            raise FileNotFoundError(
                f"log_returns is None and returns_path does not exist: {returns_path}"
            )
        log_returns = pd.read_parquet(returns_path)

    # Slice the calm period from the returns frame (half-open on the right
    # so we strictly stop before the precursor signal at 17:14 UTC).
    calm = log_returns.loc[
        (log_returns.index >= CALM_START) & (log_returns.index < CALM_END)
    ]
    if calm.empty:
        raise ValueError(
            f"calm period {CALM_START} .. {CALM_END} produced 0 rows from log_returns."
        )
    if calm.shape[0] <= WINDOW_SIZE + INDICATOR_WINDOW + SURROGATE_BASELINE_HEAD_MINUTES:
        raise ValueError(
            f"calm period has only {calm.shape[0]} rows; need at least "
            f"WINDOW_SIZE+INDICATOR_WINDOW+head={WINDOW_SIZE + INDICATOR_WINDOW + SURROGATE_BASELINE_HEAD_MINUTES} "
            "for the T_run_max statistic to be well-defined."
        )

    # Observed side: run the Session 8 pipeline on the *full* 5-day frame
    # so we have a contiguous L^1 H_1 series that covers both the baseline
    # window (Oct 8 00:00 – 03:00 UTC) and the precursor window
    # (Oct 10 17:14 – 21:14 UTC).
    logger.info(
        "computing observed L^1 H_1 on full frame (%d rows) for baseline + precursor",
        log_returns.shape[0],
    )
    real_l1_h1 = _real_l1_h1_series(log_returns)
    observed_baseline = _baseline_median_from_real_series(real_l1_h1)
    observed_t, observed_count = _observed_t_run_max(
        real_l1_h1, baseline=observed_baseline
    )
    logger.info(
        "observed baseline=%.4g, T_run_max=%.4f h (%d minutes of rolling mean > %.4g)",
        observed_baseline,
        observed_t,
        observed_count,
        RUN_MULTIPLIER * observed_baseline,
    )

    returns_2d = calm.to_numpy()
    phase_result = _run_method(
        returns_2d,
        method="phase_random",
        n_surrogates=n_surrogates,
        seed=seed,
        observed_t_run_max=observed_t,
        n_jobs=n_jobs,
    )
    bootstrap_result = _run_method(
        returns_2d,
        method="bootstrap",
        n_surrogates=n_surrogates,
        seed=seed + 1,  # distinct seed so the two nulls do not share randomness
        observed_t_run_max=observed_t,
        n_jobs=n_jobs,
    )

    result = Session9Result(
        phase=phase_result,
        bootstrap=bootstrap_result,
        observed_baseline=observed_baseline,
        observed_t_run_max=observed_t,
        observed_precursor_above_count=observed_count,
        observed_l1_h1=real_l1_h1,
        n_surrogates=n_surrogates,
    )

    if figure_path is not None:
        result.figure_path = plot_surrogate_test(
            result=result,
            figure_path=figure_path,
            calm_index=calm.index,
            n_display=n_display,
        )
    if report_path is not None:
        result.report_path = write_surrogate_findings_report(
            result=result, report_path=report_path
        )
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _hist_panel(
    ax: plt.Axes,
    *,
    title: str,
    null: np.ndarray,
    observed: float,
    p_value: float,
    color: str,
) -> None:
    finite = null[np.isfinite(null)]
    n_bins = max(10, min(40, int(np.sqrt(finite.size)) + 1)) if finite.size else 10
    if finite.size:
        ax.hist(finite, bins=n_bins, color=color, alpha=0.75, edgecolor="black", lw=0.4)
    ax.axvline(
        observed,
        color="red",
        lw=1.5,
        ls="--",
        label=f"observed T = {observed:.3f} h",
    )
    ax.set_xlabel(r"$T_{\mathrm{run\_max}}$ (hours of 60-min rolling mean > 7x baseline)")
    ax.set_ylabel("count")
    ax.set_title(f"{title}  (p = {p_value:.3g})", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)


def _overlay_panel(
    ax: plt.Axes,
    *,
    observed_series: pd.Series,
    calm_index: pd.DatetimeIndex,
    phase_l1_h1: np.ndarray,
    n_display: int,
    seed: int,
) -> None:
    """Overlay ``n_display`` random phase-randomized L^1 H_1 traces on the observed."""
    rng = np.random.default_rng(seed)
    n_avail = phase_l1_h1.shape[0]
    pick = rng.choice(n_avail, size=min(n_display, n_avail), replace=False)

    # Map calm-period window-end times: row k of the L^1 H_1 array corresponds
    # to calm_index[WINDOW_SIZE - 1 + k]. The calm period starts at 00:01 UTC
    # (first diff in the returns frame), so this gives correct time labels.
    n_windows = phase_l1_h1.shape[1]
    surrogate_index = calm_index[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + n_windows]

    # Floor below which log-axis would blow up; pick smallest positive value
    # across the displayed traces and the observed series.
    floor_candidates: list[float] = []
    for s_idx in pick:
        arr = phase_l1_h1[s_idx]
        pos = arr[arr > 0]
        if pos.size:
            floor_candidates.append(float(pos.min()))
    pos_obs = observed_series[observed_series > 0]
    if not pos_obs.empty:
        floor_candidates.append(float(pos_obs.min()))
    floor = max(min(floor_candidates) if floor_candidates else 1e-12, 1e-12)

    for s_idx in pick:
        arr = np.maximum(phase_l1_h1[s_idx], floor)
        ax.plot(surrogate_index, arr, color="0.7", lw=0.4, alpha=0.6)
    obs_arr = np.maximum(observed_series.to_numpy(), floor)
    ax.plot(observed_series.index, obs_arr, color="red", lw=0.9, label="observed")
    ax.axvline(CASCADE_CENTER, color="black", lw=0.7, ls=":", alpha=0.7, label="cascade")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\|\lambda\|_1^{H_1}$ (log)")
    ax.set_xlabel("time (UTC)")
    ax.set_title(
        f"L^1 H_1: observed (red) vs {len(pick)} phase-randomized surrogates "
        "(grey)",
        fontsize=10,
    )
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.legend(loc="upper left", fontsize=9)


def plot_surrogate_test(
    *,
    result: Session9Result,
    figure_path: Path,
    calm_index: pd.DatetimeIndex,
    n_display: int = DEFAULT_N_DISPLAY,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Three-panel figure: phase-random hist, bootstrap hist, L^1 H_1 overlay."""
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    _hist_panel(
        axes[0],
        title=f"Phase-randomized null  (N = {result.n_surrogates})",
        null=result.phase.t_run_max,
        observed=result.observed_t_run_max,
        p_value=result.phase.p_value,
        color="#4477AA",
    )
    _hist_panel(
        axes[1],
        title=f"Bootstrap null  (N = {result.n_surrogates})",
        null=result.bootstrap.t_run_max,
        observed=result.observed_t_run_max,
        p_value=result.bootstrap.p_value,
        color="#CC6677",
    )
    _overlay_panel(
        axes[2],
        observed_series=result.observed_l1_h1,
        calm_index=calm_index,
        phase_l1_h1=result.phase.l1_h1_array,
        n_display=n_display,
        seed=seed,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)
    return figure_path


# ---------------------------------------------------------------------------
# Findings report
# ---------------------------------------------------------------------------


def _safe_relpath(path: Optional[Path]) -> str:
    if path is None:
        return "n/a"
    try:
        return str(Path(path).relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _quantile_table(t_run: np.ndarray) -> dict[str, float]:
    finite = t_run[np.isfinite(t_run)]
    if finite.size == 0:
        return {"q05": float("nan"), "q50": float("nan"), "q95": float("nan"), "max": float("nan")}
    return {
        "q05": float(np.quantile(finite, 0.05)),
        "q50": float(np.quantile(finite, 0.50)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def _verdict(p_value: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p_value):
        return "INCONCLUSIVE (p is NaN)"
    return "SIGNIFICANT at p<0.05" if p_value < alpha else "NOT significant at p<0.05"


def write_surrogate_findings_report(
    *, result: Session9Result, report_path: Path
) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    phase_q = _quantile_table(result.phase.t_run_max)
    boot_q = _quantile_table(result.bootstrap.t_run_max)

    threshold = RUN_MULTIPLIER * result.observed_baseline

    lines = [
        "# Surrogate test of the October 10, 2025 L^1 H_1 precursor",
        "",
        "**Session 9 — null-distribution test of the Session 8 precursor finding.** "
        "Session 8 reported a 4 – 6 hour pre-cascade build-up in "
        r"$\|\lambda\|_1^{H_1}$"
        " at minute resolution: peak 16.95x baseline at 2025-10-10 17:14 UTC "
        "(241 minutes before the cascade), with hourly means 3 – 7x baseline "
        "across the 15:15 – 21:15 UTC window. The open question was whether "
        "this elevation is statistically distinguishable from noise, or "
        "whether a 4 – 6 hour stretch above 7x baseline can be produced by "
        "chance in calm data.",
        "",
        "## Setup",
        "",
        f"* Calm period: {CALM_START.strftime('%Y-%m-%d %H:%M UTC')} .. "
        f"{CALM_END.strftime('%Y-%m-%d %H:%M UTC')} "
        f"(~{(CALM_END - CALM_START).total_seconds() / 3600:.0f} hours of pre-precursor data).",
        f"* Observed precursor window: {PRECURSOR_START.strftime('%Y-%m-%d %H:%M UTC')} .. "
        f"{PRECURSOR_END.strftime('%Y-%m-%d %H:%M UTC')} "
        f"(strictly before cascade center {CASCADE_CENTER.strftime('%H:%M UTC')}).",
        f"* TDA pipeline: `multivariate`, W = {WINDOW_SIZE} min, "
        f"$r_{{\\max}} = {MAX_EDGE_LENGTH}$, $H_0$ and $H_1$ (Session 8 settings).",
        f"* Statistic: $T_{{\\mathrm{{run\\_max}}}}$ = longest run (in fractional hours) "
        f"of the {INDICATOR_WINDOW}-minute rolling mean of "
        r"$\|\lambda\|_1^{H_1}$"
        f" above ${RUN_MULTIPLIER:g} \\times$ baseline.",
        f"* Surrogate baseline: median over first "
        f"{SURROGATE_BASELINE_HEAD_MINUTES} indicator-window samples (= 3 hours).",
        f"* Observed baseline: Session 8 value (median over "
        f"{BASELINE_START.strftime('%Y-%m-%d %H:%M UTC')} .. "
        f"{BASELINE_END.strftime('%H:%M UTC')}) = "
        f"{result.observed_baseline:.4g}.",
        f"* $7 \\times$ observed baseline = {threshold:.4g}.",
        f"* Number of surrogates per method: {result.n_surrogates}.",
        f"* Symbols: {', '.join(SYMBOLS_OCT10)} (six majors).",
        "",
        "## Observed",
        "",
        f"* **Observed $T_{{\\mathrm{{run\\_max}}}}$:** "
        f"{result.observed_t_run_max:.4f} hours "
        f"({int(round(result.observed_t_run_max * 60))} consecutive minutes of "
        f"rolling mean above 7x baseline within the 17:14 – 21:14 UTC window).",
        f"* Total minutes in the precursor window where the rolling mean "
        f"exceeds threshold: {result.observed_precursor_above_count} / "
        f"{int((PRECURSOR_END - PRECURSOR_START).total_seconds() / 60) + 1}.",
        "",
        "## Surrogate quantiles",
        "",
        "| method | $q_{0.05}$ | $q_{0.50}$ | $q_{0.95}$ | max | p-value | verdict |",
        "|---|---|---|---|---|---|---|",
        (
            f"| Phase-randomized (Prichard-Theiler multivariate) | "
            f"{phase_q['q05']:.3f} h | {phase_q['q50']:.3f} h | "
            f"{phase_q['q95']:.3f} h | {phase_q['max']:.3f} h | "
            f"{result.phase.p_value:.3g} | {_verdict(result.phase.p_value)} |"
        ),
        (
            f"| Moving-block bootstrap (Kuensch 1989) | "
            f"{boot_q['q05']:.3f} h | {boot_q['q50']:.3f} h | "
            f"{boot_q['q95']:.3f} h | {boot_q['max']:.3f} h | "
            f"{result.bootstrap.p_value:.3g} | {_verdict(result.bootstrap.p_value)} |"
        ),
        "",
        "## Reading the table",
        "",
        "* The **phase-randomized null** preserves each column's power "
        "spectrum *and* the cross-spectrum between symbols (Prichard & "
        "Theiler 1994 shared-phase trick). This is the demanding null for "
        "a multivariate TDA pipeline because lagged cross-correlations "
        "between BTC/ETH/SOL/BNB/XRP/DOGE are exactly the structure the "
        "point cloud sees.",
        "* The **bootstrap null** (moving-block, Politis-White block "
        "length rule) preserves the marginal distribution exactly but "
        "destroys long-range serial dependence. It is the less strict "
        "comparison and reproduces the null used by Ismail et al. (2020).",
        "* iAAFT is *not* used because there is no consensus multivariate "
        "formulation that preserves cross-spectra and marginals jointly; "
        "applying iAAFT independently per column would destroy the "
        "cross-correlations driving the multivariate Vietoris-Rips "
        "topology.",
        "",
        "## Verdict",
        "",
        (
            f"Under phase randomization the precursor is "
            f"**{_verdict(result.phase.p_value)}** "
            f"(empirical p = {result.phase.p_value:.3g})."
        ),
        (
            f"Under moving-block bootstrap the precursor is "
            f"**{_verdict(result.bootstrap.p_value)}** "
            f"(empirical p = {result.bootstrap.p_value:.3g})."
        ),
        "",
        "## Caveats",
        "",
        "* The conservative North et al. (2002) p-value convention "
        f"$p = (1 + k) / (1 + N)$ gives a smallest reportable p of "
        f"$1 / (1 + {result.n_surrogates}) = "
        f"{1.0 / (1.0 + result.n_surrogates):.3g}$. p-values at this floor "
        "mean *no* surrogate equalled or exceeded the observed statistic.",
        "* The 60-minute rolling mean threshold is a *single* exceedance "
        "criterion. The Session 8 hourly table shows the per-hour means "
        "vary across 3 – 7x baseline within the 15:15 – 21:15 UTC window "
        f"-- the 7x criterion only triggers in the densest sub-windows. "
        f"Observed $T_{{\\mathrm{{run\\_max}}}}$ = "
        f"{result.observed_t_run_max:.3f} h "
        f"({int(round(result.observed_t_run_max * 60))} minutes) is the "
        "longest such densely-exceeding stretch within the strict "
        "pre-cascade window.",
        "* The surrogate window is ~64 hours; the observed precursor "
        "window is ~4 hours. The surrogate has 16x more opportunities to "
        "produce a long run, so the test is conservative (biased against "
        "rejecting the null).",
        "* The two daily-resolution null tests (Terra-Luna, FTX) have "
        "not been re-run at minute resolution yet — that is Session 10.",
        "",
        "## Outputs",
        "",
        f"* Figure: `{_safe_relpath(result.figure_path)}`",
        f"* This report: `{_safe_relpath(report_path)}`",
        "",
    ]
    text = "\n".join(lines) + "\n"
    report_path.write_text(text)
    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> int:  # pragma: no cover - exercised by hand
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_session9_surrogate_analysis()
    print()
    print(f"Figure: {result.figure_path}")
    print(f"Report: {result.report_path}")
    print(f"Phase-random p-value:  {result.phase.p_value:.3g}")
    print(f"Bootstrap   p-value:   {result.bootstrap.p_value:.3g}")
    print(f"Observed T_run_max:    {result.observed_t_run_max:.4f} h")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_cli())
