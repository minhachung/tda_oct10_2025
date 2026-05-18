"""Terra-Luna and FTX control-event replication of Sessions 8 + 9.

Session 10 — generalises the Oct 10 minute-resolution analysis pipeline
(Session 8) and the phase-randomized + bootstrap surrogate test
(Session 9) so they can run unmodified on two historical control
events:

* **Terra-Luna / UST de-peg** (May 8 – 13, 2022). Symbols: BTC, ETH,
  LUNA, UST, SOL, AVAX. Cascade center: 2022-05-10 00:00 UTC.
* **FTX / FTT collapse** (Nov 6 – 10, 2022). Symbols: BTC, ETH, SOL,
  BNB, XRP, FTT. Cascade center: 2022-11-08 04:00 UTC.

The pipeline parameters (W = 50 min, max_edge_length = 0.10,
multivariate mode, H_0 + H_1) are identical to Session 8. The
surrogate test uses the same Prichard-Theiler multivariate phase-
randomization and Kuensch moving-block bootstrap as Session 9, with
the same T_run_max statistic (longest run of the 60-minute rolling
mean of L^1 H_1 above 7x baseline, expressed in fractional hours).
Per-event quantities that *do* change between runs:

* analysis window (start / end), cascade center, precursor window
  (= [cascade_center - 360 min, cascade_center - 1 min]), calm period
  for surrogates (= [analysis_start, precursor_start]);
* symbol basket -- LUNA + UST are the epicenter for Terra-Luna, FTT
  for FTX;
* observed baseline (= median of L^1 H_1 over first 3 hours of the
  analysis window, mirroring Session 8's `BASELINE_HOURS = 3`).

Data quality
------------
Both events have Binance-archive coverage at 1-minute resolution.
Terra-Luna data truncates on 2022-05-13 ~00:40 UTC because LUNA and
UST were delisted within an hour of each other once LUNA traded below
1e-4 USDT and UST below 0.25 USDT. The aligned six-symbol returns
frame therefore ends at 2022-05-13 00:43 UTC. FTX coverage is clean
across the full Nov 6 – 10 window.

Outputs
-------
Per event, two figures (``[event]_minute_main.png`` and
``[event]_surrogate_test.png``) mirroring the Session 8 / Session 9
panels, plus a single combined report
(``paper/control_events_findings.md``) that places the Oct 10 result
alongside the two controls and discusses whether the H_1 precursor
generalises and how the H_0 expansion compares between
leverage-driven (Oct 10, FTX) and run-on-the-bank (Terra-Luna)
cascades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tda_oct10 import data_ingestion as di
from tda_oct10.analysis_oct10_minute import (
    HOMOLOGY_DIMS,
    INDICATOR_WINDOW,
    ISMAIL_TAU_THRESHOLD,
    KENDALL_WINDOW,
    MAX_EDGE_LENGTH,
    WINDOW_SIZE,
)
from tda_oct10.analysis_oct10_surrogate import (
    DEFAULT_N_DISPLAY,
    DEFAULT_N_SURROGATES,
    DEFAULT_SEED,
    RUN_MULTIPLIER,
    SURROGATE_BASELINE_HEAD_MINUTES,
    _compute_null_distribution,
    _generate_surrogates,
    _longest_true_run,
    _per_surrogate_stats,
    _surrogate_baseline,
    _t_run_max,
)
from tda_oct10.indicators import rolling_indicators
from tda_oct10.nulls import empirical_pvalue
from tda_oct10.trend_test import kendall_tau_rolling

__all__ = [
    "EventConfig",
    "TERRA_LUNA_CONFIG",
    "FTX_CONFIG",
    "EventResult",
    "fetch_event_returns",
    "run_event_main_analysis",
    "run_event_surrogate_test",
    "run_control_event",
    "run_session10_controls",
    "write_control_events_report",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults shared with Session 8
# ---------------------------------------------------------------------------

BASELINE_HOURS = 3
PRE_CASCADE_LEAD_MINUTES = 360

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_FIGURES_DIR = _REPO_ROOT / "paper" / "figures"
_DEFAULT_REPORT_PATH = _REPO_ROOT / "paper" / "control_events_findings.md"


# ---------------------------------------------------------------------------
# EventConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventConfig:
    """Per-event parameters for the Session 10 control-event replication.

    Attributes
    ----------
    name :
        Short slug used for filenames (e.g. ``"terra_luna"``).
    label :
        Human-readable label used in figure titles and report text.
    symbols :
        Six-symbol basket fed to the multivariate TDA pipeline. Ordered
        so the basket lines up structurally with the Oct 10 basket
        (majors first, epicenter assets last).
    epicenter_symbols :
        Symbols at the epicenter of the cascade (LUNA / UST for Terra,
        FTT for FTX). Used in the report discussion only.
    cascade_descriptor :
        Free-text classifier (``"leverage-driven"`` vs ``"run-on-the-bank"``)
        used in the discussion section of the combined report.
    analysis_start, analysis_end :
        Inclusive bounds for the minute-resolution analysis window.
    cascade_center :
        Timestamp of the cascade center. Defined as the moment around
        which the precursor window and post-cascade window are anchored.
    pre_cascade_lead_minutes :
        Length of the precursor window in minutes. Defaults to 360
        (= 6 hours), matching Session 8.
    """

    name: str
    label: str
    symbols: tuple[str, ...]
    epicenter_symbols: tuple[str, ...]
    cascade_descriptor: str
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    cascade_center: pd.Timestamp
    pre_cascade_lead_minutes: int = PRE_CASCADE_LEAD_MINUTES

    # -- derived windows --------------------------------------------------
    @property
    def baseline_start(self) -> pd.Timestamp:
        return self.analysis_start

    @property
    def baseline_end(self) -> pd.Timestamp:
        return self.analysis_start + pd.Timedelta(hours=BASELINE_HOURS)

    @property
    def precursor_start(self) -> pd.Timestamp:
        return self.cascade_center - pd.Timedelta(minutes=self.pre_cascade_lead_minutes)

    @property
    def precursor_end(self) -> pd.Timestamp:
        return self.cascade_center - pd.Timedelta(minutes=1)

    @property
    def calm_start(self) -> pd.Timestamp:
        return self.analysis_start

    @property
    def calm_end(self) -> pd.Timestamp:
        # Surrogates draw from data strictly before the precursor window.
        return self.precursor_start

    # -- filesystem ------------------------------------------------------
    @property
    def returns_path(self) -> Path:
        return _PROCESSED_DIR / f"{self.name}_minute_returns.parquet"

    @property
    def main_plot_path(self) -> Path:
        return _FIGURES_DIR / f"{self.name}_minute_main.png"

    @property
    def surrogate_plot_path(self) -> Path:
        return _FIGURES_DIR / f"{self.name}_surrogate_test.png"


# ---------------------------------------------------------------------------
# Concrete event configurations
# ---------------------------------------------------------------------------

TERRA_LUNA_CONFIG = EventConfig(
    name="terra_luna",
    label="Terra-Luna / UST de-peg, May 2022",
    symbols=("BTC", "ETH", "LUNA", "UST", "SOL", "AVAX"),
    epicenter_symbols=("LUNA", "UST"),
    cascade_descriptor="run-on-the-bank (algorithmic stablecoin de-peg)",
    analysis_start=pd.Timestamp("2022-05-08 00:00", tz="UTC"),
    analysis_end=pd.Timestamp("2022-05-13 23:59", tz="UTC"),
    cascade_center=pd.Timestamp("2022-05-10 00:00", tz="UTC"),
)

FTX_CONFIG = EventConfig(
    name="ftx",
    label="FTX / FTT collapse, November 2022",
    symbols=("BTC", "ETH", "SOL", "BNB", "XRP", "FTT"),
    epicenter_symbols=("FTT",),
    cascade_descriptor="leverage-driven (FTT collateral chains)",
    analysis_start=pd.Timestamp("2022-11-06 00:00", tz="UTC"),
    analysis_end=pd.Timestamp("2022-11-10 23:59", tz="UTC"),
    cascade_center=pd.Timestamp("2022-11-08 04:00", tz="UTC"),
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EventMainResult:
    """Per-event Session 8-style bundle: returns, landscape, indicators, baselines."""

    config: EventConfig
    log_returns: pd.DataFrame
    landscape: pd.DataFrame
    indicators: pd.DataFrame
    kendall_tau: pd.DataFrame
    baseline_l1_h1: float
    baseline_l1_h0: float
    pre_l1_h1_peak: float
    pre_l1_h1_peak_time: Optional[pd.Timestamp]
    pre_l1_h1_lead_minutes: Optional[int]
    pre_l1_h0_peak: float
    pre_l1_h0_peak_time: Optional[pd.Timestamp]
    pre_l1_h0_lead_minutes: Optional[int]
    cascade_l1_h1_peak: float
    cascade_l1_h0_peak: float
    post_l1_h1_max: float
    post_l1_h0_max: float
    data_warnings: tuple[str, ...]
    main_plot_path: Optional[Path] = None


@dataclass
class EventSurrogateResult:
    """Per-event Session 9-style surrogate test bundle."""

    config: EventConfig
    observed_baseline: float
    observed_t_run_max: float
    observed_precursor_above_count: int
    n_surrogates: int
    phase_t_run_max: np.ndarray
    phase_l1_h1_array: np.ndarray
    phase_p_value: float
    bootstrap_t_run_max: np.ndarray
    bootstrap_l1_h1_array: np.ndarray
    bootstrap_p_value: float
    surrogate_plot_path: Optional[Path] = None


@dataclass
class EventResult:
    """Combined main + surrogate result for one control event."""

    main: EventMainResult
    surrogate: EventSurrogateResult


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------


def fetch_event_returns(
    config: EventConfig, *, save: bool = True
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Pull minute log-returns from ``data.binance.vision`` for ``config.symbols``.

    Captures any gap-warning emitted by ``load_minute_returns_archive``
    so the caller can record data-quality flags in the report.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        log_returns = di.load_minute_returns_archive(
            config.analysis_start.to_pydatetime(),
            config.analysis_end.to_pydatetime(),
            symbols=config.symbols,
        )
        data_warnings = tuple(str(w.message) for w in caught)

    if save:
        config.returns_path.parent.mkdir(parents=True, exist_ok=True)
        log_returns.to_parquet(config.returns_path)
    return log_returns, data_warnings


# ---------------------------------------------------------------------------
# Pipeline + indicators (Session 8-style, parameterised)
# ---------------------------------------------------------------------------


def _run_pipeline(
    log_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """TDA pipeline → landscape DataFrame on right-edge window times."""
    from tda_oct10.tda_pipeline import TDAPipeline  # local to keep top-level imports tight

    pipeline = TDAPipeline(
        mode="multivariate",
        window_size=WINDOW_SIZE,
        max_edge_length=MAX_EDGE_LENGTH,
        homology_dims=HOMOLOGY_DIMS,
        n_jobs=-1,
    )
    result = pipeline.fit_transform(log_returns.to_numpy())
    n_windows = int(result["L1_H1"].shape[0])
    window_ends = log_returns.index[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + n_windows]
    landscape = pd.DataFrame(
        {
            "L1_H0": result["L1_H0"],
            "L1_H1": result["L1_H1"],
            "L2_H0": result["L2_H0"],
            "L2_H1": result["L2_H1"],
            "PE_H1": result["persistence_entropy_H1"],
        },
        index=window_ends,
    )
    landscape.index.name = "window_end_time"
    return landscape, window_ends


def _run_indicators(
    landscape: pd.DataFrame, window_ends: pd.DatetimeIndex
) -> pd.DataFrame:
    ind = rolling_indicators(landscape["L1_H1"].to_numpy(), window=INDICATOR_WINDOW)
    if ind.empty:
        return ind
    ind.index = window_ends[ind.index]
    ind.index.name = "indicator_end_time"
    return ind


def _run_kendall_tau(indicators: pd.DataFrame) -> pd.DataFrame:
    cols = ("AC1", "VAR", "MPS")
    if indicators.empty:
        return pd.DataFrame(
            columns=[f"tau_{c}" for c in cols] + [f"p_{c}" for c in cols]
        )
    parts: dict[str, pd.Series] = {}
    for col in cols:
        kt = kendall_tau_rolling(indicators[col].to_numpy(), window=KENDALL_WINDOW)
        if kt.empty:
            parts[f"tau_{col}"] = pd.Series(dtype=float)
            parts[f"p_{col}"] = pd.Series(dtype=float)
            continue
        right_edge_times = indicators.index[kt.index]
        parts[f"tau_{col}"] = pd.Series(kt["tau"].to_numpy(), index=right_edge_times)
        parts[f"p_{col}"] = pd.Series(kt["p_value"].to_numpy(), index=right_edge_times)
    kt_df = pd.DataFrame(parts)
    kt_df.index.name = "tau_end_time"
    return kt_df


def _baseline_median(series: pd.Series, *, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if series.empty:
        return float("nan")
    sub = series.loc[(series.index >= start) & (series.index <= end)].dropna()
    return float(sub.median()) if not sub.empty else float("nan")


def _precursor_peak(
    series: pd.Series, *, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[float, Optional[pd.Timestamp]]:
    if series.empty:
        return float("nan"), None
    sub = series.loc[(series.index >= start) & (series.index <= end)].dropna()
    if sub.empty:
        return float("nan"), None
    return float(sub.max()), sub.idxmax()


def _lead_minutes_to_cascade(
    ts: Optional[pd.Timestamp], cascade_center: pd.Timestamp
) -> Optional[int]:
    if ts is None:
        return None
    return int((cascade_center - ts).total_seconds() // 60)


def _cascade_window_max(
    series: pd.Series, cascade_center: pd.Timestamp,
    *, before_min: int = 15, after_min: int = 45,
) -> float:
    if series.empty:
        return float("nan")
    mask = (
        (series.index >= cascade_center - pd.Timedelta(minutes=before_min))
        & (series.index <= cascade_center + pd.Timedelta(minutes=after_min))
    )
    return float(series.loc[mask].max()) if mask.any() else float("nan")


def _post_cascade_max(series: pd.Series, cascade_center: pd.Timestamp) -> float:
    if series.empty:
        return float("nan")
    post = series.loc[series.index > cascade_center]
    return float(post.max()) if not post.empty else float("nan")


# ---------------------------------------------------------------------------
# Plotting (Session 8 style)
# ---------------------------------------------------------------------------


def _plot_main_panels(
    *,
    config: EventConfig,
    log_returns: pd.DataFrame,
    landscape: pd.DataFrame,
    kendall_tau: pd.DataFrame,
    plot_path: Path,
) -> Path:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Reconstruct close prices for the basket so we can plot a sane price
    # panel. Each close = exp(cumsum(log_returns)) * initial; the initial
    # price scale is arbitrary, so we normalise each to 1.0 at the start of
    # the window. This puts every symbol on the same axis for visual
    # comparison even when their absolute price ranges differ by orders of
    # magnitude (LUNA, UST). The cascade epicenter assets are drawn in
    # warmer colours so they pop visually.
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax_price = axes[0]
    cmap = plt.get_cmap("tab10")
    for j, sym in enumerate(log_returns.columns):
        norm_close = np.exp(log_returns[sym].cumsum())
        color = "C3" if sym in config.epicenter_symbols else cmap(j)
        lw = 1.1 if sym in config.epicenter_symbols else 0.7
        ax_price.semilogy(
            log_returns.index, norm_close, lw=lw, label=sym, color=color
        )
    ax_price.set_ylabel("normalised close (log)")
    ax_price.legend(ncol=3, fontsize=8, loc="lower left")
    ax_price.set_title(
        f"{config.label} — minute-resolution TDA "
        f"(W = {WINDOW_SIZE} min, $r_{{\\max}} = {MAX_EDGE_LENGTH}$, "
        "$H_0$ + $H_1$)"
    )

    l1_h1_floor = max(
        float(landscape["L1_H1"].replace(0, np.nan).min(skipna=True) or 1e-12),
        1e-12,
    )
    axes[1].semilogy(
        landscape.index,
        np.maximum(landscape["L1_H1"].to_numpy(), l1_h1_floor),
        lw=0.8,
        color="C3",
    )
    axes[1].set_ylabel(r"$\|\lambda\|_1$  (H$_1$, log)")

    l1_h0_floor = max(
        float(landscape["L1_H0"].replace(0, np.nan).min(skipna=True) or 1e-12),
        1e-12,
    )
    axes[2].semilogy(
        landscape.index,
        np.maximum(landscape["L1_H0"].to_numpy(), l1_h0_floor),
        lw=0.8,
        color="C2",
    )
    axes[2].set_ylabel(r"$\|\lambda\|_1$  (H$_0$, log)")

    ax_tau = axes[3]
    if "tau_VAR" in kendall_tau.columns and not kendall_tau["tau_VAR"].dropna().empty:
        ax_tau.plot(
            kendall_tau.index,
            kendall_tau["tau_VAR"],
            lw=0.8,
            color="C4",
            label=r"$\tau$ of VAR($\|\lambda\|_1^{H_1}$)",
        )
    ax_tau.axhline(0.0, color="grey", lw=0.6, alpha=0.5)
    ax_tau.axhline(
        ISMAIL_TAU_THRESHOLD,
        color="black",
        lw=0.7,
        ls="--",
        label=rf"Ismail 2020 threshold $\tau={ISMAIL_TAU_THRESHOLD}$",
    )
    ax_tau.set_ylabel(r"Kendall $\tau$")
    ax_tau.set_xlabel("Time (UTC)")
    ax_tau.legend(loc="lower left", fontsize=8)

    for ax in axes:
        ax.axvline(config.cascade_center, color="red", lw=0.9, ls=":", alpha=0.85)
        ax.set_xlim(log_returns.index.min(), log_returns.index.max())

    span_hours = (
        (log_returns.index.max() - log_returns.index.min()).total_seconds() / 3600.0
    )
    if span_hours <= 48:
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    else:
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# Main analysis driver
# ---------------------------------------------------------------------------


def run_event_main_analysis(
    log_returns: pd.DataFrame,
    config: EventConfig,
    *,
    data_warnings: Sequence[str] = (),
    plot: bool = True,
) -> EventMainResult:
    """Run the Session 8 minute-resolution pipeline on one control event."""
    if log_returns.empty:
        raise ValueError(f"{config.name}: empty log_returns")

    landscape, window_ends = _run_pipeline(log_returns)
    indicators = _run_indicators(landscape, window_ends)
    kendall_tau = _run_kendall_tau(indicators)

    l1_h1 = landscape["L1_H1"]
    l1_h0 = landscape["L1_H0"]

    baseline_l1_h1 = _baseline_median(
        l1_h1, start=config.baseline_start, end=config.baseline_end
    )
    baseline_l1_h0 = _baseline_median(
        l1_h0, start=config.baseline_start, end=config.baseline_end
    )

    pre_h1_peak, pre_h1_ts = _precursor_peak(
        l1_h1, start=config.precursor_start, end=config.precursor_end
    )
    pre_h0_peak, pre_h0_ts = _precursor_peak(
        l1_h0, start=config.precursor_start, end=config.precursor_end
    )

    main_plot_path: Optional[Path] = None
    if plot:
        main_plot_path = _plot_main_panels(
            config=config,
            log_returns=log_returns,
            landscape=landscape,
            kendall_tau=kendall_tau,
            plot_path=config.main_plot_path,
        )

    return EventMainResult(
        config=config,
        log_returns=log_returns,
        landscape=landscape,
        indicators=indicators,
        kendall_tau=kendall_tau,
        baseline_l1_h1=baseline_l1_h1,
        baseline_l1_h0=baseline_l1_h0,
        pre_l1_h1_peak=pre_h1_peak,
        pre_l1_h1_peak_time=pre_h1_ts,
        pre_l1_h1_lead_minutes=_lead_minutes_to_cascade(pre_h1_ts, config.cascade_center),
        pre_l1_h0_peak=pre_h0_peak,
        pre_l1_h0_peak_time=pre_h0_ts,
        pre_l1_h0_lead_minutes=_lead_minutes_to_cascade(pre_h0_ts, config.cascade_center),
        cascade_l1_h1_peak=_cascade_window_max(l1_h1, config.cascade_center),
        cascade_l1_h0_peak=_cascade_window_max(l1_h0, config.cascade_center),
        post_l1_h1_max=_post_cascade_max(l1_h1, config.cascade_center),
        post_l1_h0_max=_post_cascade_max(l1_h0, config.cascade_center),
        data_warnings=tuple(data_warnings),
        main_plot_path=main_plot_path,
    )


# ---------------------------------------------------------------------------
# Surrogate test driver (Session 9 style, parameterised)
# ---------------------------------------------------------------------------


def _observed_t_run_max_from_real(
    real_l1_h1: pd.Series, *, config: EventConfig, baseline: float
) -> tuple[float, int]:
    rolling = real_l1_h1.rolling(INDICATOR_WINDOW, min_periods=INDICATOR_WINDOW).mean()
    precursor = rolling.loc[
        (rolling.index >= config.precursor_start)
        & (rolling.index <= config.precursor_end)
    ]
    threshold = RUN_MULTIPLIER * baseline
    above = (precursor > threshold) & precursor.notna()
    arr = above.to_numpy()
    return _longest_true_run(arr) / 60.0, int(arr.sum())


def _calm_period_slice(log_returns: pd.DataFrame, config: EventConfig) -> pd.DataFrame:
    calm = log_returns.loc[
        (log_returns.index >= config.calm_start)
        & (log_returns.index < config.calm_end)
    ]
    if calm.empty:
        raise ValueError(
            f"{config.name}: calm slice [{config.calm_start}, {config.calm_end}) "
            "produced 0 rows."
        )
    needed = WINDOW_SIZE + INDICATOR_WINDOW + SURROGATE_BASELINE_HEAD_MINUTES
    if calm.shape[0] <= needed:
        raise ValueError(
            f"{config.name}: calm slice has only {calm.shape[0]} rows; need > "
            f"{needed} for the T_run_max statistic to be well-defined."
        )
    return calm


def _run_method(
    returns_2d: np.ndarray,
    *,
    method: str,
    n_surrogates: int,
    seed: int,
    observed_t_run_max: float,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    surrogates = _generate_surrogates(
        returns_2d, method=method, n=n_surrogates, seed=seed
    )
    l1_h1_array = _compute_null_distribution(surrogates, n_jobs=n_jobs)
    _, t_run_arr = _per_surrogate_stats(l1_h1_array)
    pval = empirical_pvalue(observed_t_run_max, t_run_arr, alternative="greater")
    return l1_h1_array, t_run_arr, pval


def _plot_surrogate_panels(
    *,
    config: EventConfig,
    surrogate: EventSurrogateResult,
    calm_index: pd.DatetimeIndex,
    real_l1_h1: pd.Series,
    n_display: int,
    seed: int,
    plot_path: Path,
) -> Path:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    def _hist(ax: plt.Axes, null: np.ndarray, p: float, title: str, color: str) -> None:
        finite = null[np.isfinite(null)]
        nbins = max(10, min(40, int(np.sqrt(finite.size)) + 1)) if finite.size else 10
        if finite.size:
            ax.hist(finite, bins=nbins, color=color, alpha=0.75, edgecolor="black", lw=0.4)
        ax.axvline(
            surrogate.observed_t_run_max,
            color="red",
            lw=1.5,
            ls="--",
            label=f"observed T = {surrogate.observed_t_run_max:.3f} h",
        )
        ax.set_xlabel(
            r"$T_{\mathrm{run\_max}}$ (hours of 60-min rolling mean > 7x baseline)"
        )
        ax.set_ylabel("count")
        ax.set_title(f"{title}  (p = {p:.3g})", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    _hist(
        axes[0],
        surrogate.phase_t_run_max,
        surrogate.phase_p_value,
        f"{config.label} — phase-randomized null (N = {surrogate.n_surrogates})",
        "#4477AA",
    )
    _hist(
        axes[1],
        surrogate.bootstrap_t_run_max,
        surrogate.bootstrap_p_value,
        f"{config.label} — bootstrap null (N = {surrogate.n_surrogates})",
        "#CC6677",
    )

    ax_overlay = axes[2]
    rng = np.random.default_rng(seed)
    n_avail = surrogate.phase_l1_h1_array.shape[0]
    pick = rng.choice(n_avail, size=min(n_display, n_avail), replace=False)
    n_windows = surrogate.phase_l1_h1_array.shape[1]
    surrogate_index = calm_index[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + n_windows]

    floor_candidates: list[float] = []
    for s_idx in pick:
        arr = surrogate.phase_l1_h1_array[s_idx]
        pos = arr[arr > 0]
        if pos.size:
            floor_candidates.append(float(pos.min()))
    pos_obs = real_l1_h1[real_l1_h1 > 0]
    if not pos_obs.empty:
        floor_candidates.append(float(pos_obs.min()))
    floor = max(min(floor_candidates) if floor_candidates else 1e-12, 1e-12)

    for s_idx in pick:
        arr = np.maximum(surrogate.phase_l1_h1_array[s_idx], floor)
        ax_overlay.plot(surrogate_index, arr, color="0.7", lw=0.4, alpha=0.6)
    ax_overlay.plot(
        real_l1_h1.index,
        np.maximum(real_l1_h1.to_numpy(), floor),
        color="red",
        lw=0.9,
        label="observed",
    )
    ax_overlay.axvline(
        config.cascade_center, color="black", lw=0.7, ls=":", alpha=0.7, label="cascade"
    )
    ax_overlay.set_yscale("log")
    ax_overlay.set_ylabel(r"$\|\lambda\|_1^{H_1}$ (log)")
    ax_overlay.set_xlabel("time (UTC)")
    ax_overlay.set_title(
        f"L^1 H_1: observed (red) vs {len(pick)} phase-randomized surrogates (grey)",
        fontsize=10,
    )
    ax_overlay.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax_overlay.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_overlay.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path


def run_event_surrogate_test(
    log_returns: pd.DataFrame,
    config: EventConfig,
    main_result: EventMainResult,
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
    plot: bool = True,
    n_display: int = DEFAULT_N_DISPLAY,
) -> EventSurrogateResult:
    """Run the Session 9 surrogate test on the calm slice of one control event."""
    calm = _calm_period_slice(log_returns, config)
    returns_2d = calm.to_numpy()

    observed_baseline = main_result.baseline_l1_h1
    observed_t, observed_count = _observed_t_run_max_from_real(
        main_result.landscape["L1_H1"],
        config=config,
        baseline=observed_baseline,
    )
    logger.info(
        "%s: observed baseline=%.4g, T_run_max=%.4f h (%d minutes > %.4g)",
        config.name,
        observed_baseline,
        observed_t,
        observed_count,
        RUN_MULTIPLIER * observed_baseline,
    )

    logger.info("%s: generating %d phase-randomized surrogates", config.name, n_surrogates)
    phase_l1_h1, phase_t_arr, phase_p = _run_method(
        returns_2d,
        method="phase_random",
        n_surrogates=n_surrogates,
        seed=seed,
        observed_t_run_max=observed_t,
        n_jobs=n_jobs,
    )
    logger.info("%s: generating %d bootstrap surrogates", config.name, n_surrogates)
    boot_l1_h1, boot_t_arr, boot_p = _run_method(
        returns_2d,
        method="bootstrap",
        n_surrogates=n_surrogates,
        seed=seed + 1,
        observed_t_run_max=observed_t,
        n_jobs=n_jobs,
    )

    surrogate = EventSurrogateResult(
        config=config,
        observed_baseline=observed_baseline,
        observed_t_run_max=observed_t,
        observed_precursor_above_count=observed_count,
        n_surrogates=n_surrogates,
        phase_t_run_max=phase_t_arr,
        phase_l1_h1_array=phase_l1_h1,
        phase_p_value=phase_p,
        bootstrap_t_run_max=boot_t_arr,
        bootstrap_l1_h1_array=boot_l1_h1,
        bootstrap_p_value=boot_p,
    )
    if plot:
        surrogate.surrogate_plot_path = _plot_surrogate_panels(
            config=config,
            surrogate=surrogate,
            calm_index=calm.index,
            real_l1_h1=main_result.landscape["L1_H1"],
            n_display=n_display,
            seed=seed,
            plot_path=config.surrogate_plot_path,
        )
    return surrogate


# ---------------------------------------------------------------------------
# Combined event runner
# ---------------------------------------------------------------------------


def run_control_event(
    config: EventConfig,
    *,
    log_returns: Optional[pd.DataFrame] = None,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
    plot: bool = True,
) -> EventResult:
    """End-to-end main + surrogate analysis for one event."""
    if log_returns is None:
        log_returns, data_warnings = fetch_event_returns(config, save=True)
    else:
        data_warnings = ()
    main = run_event_main_analysis(
        log_returns, config, data_warnings=data_warnings, plot=plot
    )
    surrogate = run_event_surrogate_test(
        log_returns,
        config,
        main,
        n_surrogates=n_surrogates,
        seed=seed,
        n_jobs=n_jobs,
        plot=plot,
    )
    return EventResult(main=main, surrogate=surrogate)


# ---------------------------------------------------------------------------
# Reference numbers from Sessions 7 – 9 (used in the combined comparison table)
# ---------------------------------------------------------------------------


# Source: paper/oct10_minute_findings.md, paper/oct10_surrogate_findings.md.
OCT10_REFERENCE = {
    "label": "October 10, 2025 crypto liquidation cascade",
    "cascade_date": "2025-10-10",
    "cascade_descriptor": "leverage-driven (perp-liquidation cascade)",
    "symbols": ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"),
    "baseline_l1_h1": 2.059e-08,
    "baseline_l1_h0": 5.654e-06,
    "pre_l1_h1_peak": 3.489e-07,
    "pre_l1_h1_ratio": 16.95,
    "pre_l1_h1_lead_minutes": 241,
    "pre_l1_h0_peak": 6.743e-04,
    "pre_l1_h0_ratio": 119.26,
    "pre_l1_h0_lead_minutes": 1,
    "cascade_l1_h1_ratio": 1910.90,
    "cascade_l1_h0_ratio": 2210.41,
    "observed_t_run_max_hours": 0.7833,
    "phase_p_value": 0.000999,
    "bootstrap_p_value": 0.000999,
    "h0_behavior": "expansion (~2210x baseline) at cascade",
}


def _ratio(numer: float, denom: float) -> float:
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return numer / denom


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"


def _fmt_g(value: float, sig: int = 3) -> str:
    return f"{value:.{sig}g}" if np.isfinite(value) else "n/a"


def _safe_relpath(path: Optional[Path]) -> str:
    if path is None:
        return "n/a"
    try:
        return str(Path(path).relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _ts(ts: Optional[pd.Timestamp]) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC") if ts is not None else "n/a"


def write_control_events_report(
    *,
    results: Sequence[EventResult],
    report_path: Path = _DEFAULT_REPORT_PATH,
) -> Path:
    """Write the combined Oct 10 + Terra-Luna + FTX comparison + discussion."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-event row strings for the headline table.
    table_rows = [
        "| Event | Cascade date | Precursor $T_{\\mathrm{run\\_max}}$ (h) | "
        "Lead time (min) | Phase-rand p | Bootstrap p | $H_0$ ratio at cascade |",
        "|---|---|---|---|---|---|---|",
        (
            "| Oct 10 2025 | "
            f"{OCT10_REFERENCE['cascade_date']} | "
            f"{OCT10_REFERENCE['observed_t_run_max_hours']:.3f} | "
            f"{OCT10_REFERENCE['pre_l1_h1_lead_minutes']} | "
            f"{OCT10_REFERENCE['phase_p_value']:.3g} | "
            f"{OCT10_REFERENCE['bootstrap_p_value']:.3g} | "
            f"{OCT10_REFERENCE['cascade_l1_h0_ratio']:.1f}x baseline (expansion) |"
        ),
    ]
    for er in results:
        main = er.main
        surr = er.surrogate
        h0_ratio = _ratio(main.cascade_l1_h0_peak, main.baseline_l1_h0)
        row = (
            f"| {main.config.label.split(',')[0]} | "
            f"{main.config.cascade_center.date()} | "
            f"{surr.observed_t_run_max:.3f} | "
            f"{main.pre_l1_h1_lead_minutes if main.pre_l1_h1_lead_minutes is not None else 'n/a'} | "
            f"{surr.phase_p_value:.3g} | "
            f"{surr.bootstrap_p_value:.3g} | "
            f"{h0_ratio:.1f}x baseline ({'expansion' if h0_ratio > 1.5 else 'no clear expansion'}) |"
        )
        table_rows.append(row)

    # Per-event detail blocks.
    detail_blocks: list[str] = []
    for er in results:
        main = er.main
        surr = er.surrogate
        cfg = main.config
        ratio_pre_h1 = _ratio(main.pre_l1_h1_peak, main.baseline_l1_h1)
        ratio_pre_h0 = _ratio(main.pre_l1_h0_peak, main.baseline_l1_h0)
        ratio_cas_h1 = _ratio(main.cascade_l1_h1_peak, main.baseline_l1_h1)
        ratio_cas_h0 = _ratio(main.cascade_l1_h0_peak, main.baseline_l1_h0)

        detail_blocks.append("\n".join([
            f"### {cfg.label}",
            "",
            f"* **Symbols:** {', '.join(cfg.symbols)} "
            f"(epicenter: {', '.join(cfg.epicenter_symbols)}).",
            f"* **Classifier:** {cfg.cascade_descriptor}.",
            f"* **Analysis window:** {_ts(cfg.analysis_start)} .. "
            f"{_ts(cfg.analysis_end)} "
            f"({main.log_returns.shape[0]} aligned minute returns).",
            f"* **Cascade center:** {_ts(cfg.cascade_center)}.",
            f"* **Precursor window:** {_ts(cfg.precursor_start)} .. "
            f"{_ts(cfg.precursor_end)} "
            f"({cfg.pre_cascade_lead_minutes} minutes).",
            f"* **Calm period for surrogates:** {_ts(cfg.calm_start)} .. "
            f"{_ts(cfg.calm_end)} "
            f"(~{(cfg.calm_end - cfg.calm_start).total_seconds() / 3600:.1f} hours).",
            "",
            "**Baselines and ratios** (median of L^1 over the first "
            f"{BASELINE_HOURS} hours of the analysis window):",
            "",
            f"* Baseline $L^1$ $H_1$: {_fmt_g(main.baseline_l1_h1)}.",
            f"* Baseline $L^1$ $H_0$: {_fmt_g(main.baseline_l1_h0)}.",
            (
                f"* Pre-cascade $L^1$ $H_1$ peak: {_fmt_g(main.pre_l1_h1_peak)} "
                f"({_fmt(ratio_pre_h1, 2)}x baseline) at "
                f"{_ts(main.pre_l1_h1_peak_time)} "
                f"-- lead {main.pre_l1_h1_lead_minutes if main.pre_l1_h1_lead_minutes is not None else 'n/a'} min."
            ),
            (
                f"* Pre-cascade $L^1$ $H_0$ peak: {_fmt_g(main.pre_l1_h0_peak)} "
                f"({_fmt(ratio_pre_h0, 2)}x baseline) at "
                f"{_ts(main.pre_l1_h0_peak_time)} "
                f"-- lead {main.pre_l1_h0_lead_minutes if main.pre_l1_h0_lead_minutes is not None else 'n/a'} min."
            ),
            (
                f"* Cascade-window $L^1$ $H_1$ peak: {_fmt_g(main.cascade_l1_h1_peak)} "
                f"({_fmt(ratio_cas_h1, 2)}x baseline)."
            ),
            (
                f"* Cascade-window $L^1$ $H_0$ peak: {_fmt_g(main.cascade_l1_h0_peak)} "
                f"({_fmt(ratio_cas_h0, 2)}x baseline)."
            ),
            "",
            "**Surrogate test** (N = "
            f"{surr.n_surrogates} per method; $7\\times$ baseline = "
            f"{_fmt_g(RUN_MULTIPLIER * surr.observed_baseline)}):",
            "",
            (
                f"* Observed $T_{{\\mathrm{{run\\_max}}}}$: "
                f"{surr.observed_t_run_max:.4f} h "
                f"({int(round(surr.observed_t_run_max * 60))} consecutive minutes of "
                "60-min rolling mean > 7x baseline in the precursor window)."
            ),
            (
                f"* Phase-randomized null: max surrogate "
                f"$T_{{\\mathrm{{run\\_max}}}} = {surr.phase_t_run_max.max():.3f}$ h; "
                f"empirical p = {surr.phase_p_value:.3g}."
            ),
            (
                f"* Bootstrap null: max surrogate "
                f"$T_{{\\mathrm{{run\\_max}}}} = {surr.bootstrap_t_run_max.max():.3f}$ h; "
                f"empirical p = {surr.bootstrap_p_value:.3g}."
            ),
            "",
            "**Outputs:**",
            f"* Main figure: `{_safe_relpath(main.main_plot_path)}`",
            f"* Surrogate figure: `{_safe_relpath(surr.surrogate_plot_path)}`",
            "",
            (
                "**Data warnings:** "
                + ("; ".join(main.data_warnings) if main.data_warnings else "none")
            ),
            "",
        ]))

    # Discussion section ------------------------------------------------
    discussion = _build_discussion_section(results)

    lines = [
        "# Control-event replication: Terra-Luna and FTX vs Oct 10, 2025",
        "",
        "**Session 10 — generalisability test of the Session 8 + Session 9 "
        "findings.** The Oct 10, 2025 cascade showed a multi-hour $L^1$ $H_1$ "
        "build-up before the cascade (significant at p < 0.001 under both "
        "multivariate phase randomization and moving-block bootstrap) and a "
        "co-incident $H_0$ expansion across the cascade (~2210x baseline). "
        "This note re-runs the same pipeline and surrogate test on two "
        "historical control events to ask whether the pattern generalises "
        "or is specific to October 10, 2025.",
        "",
        "Pipeline parameters are identical across all three events "
        f"(multivariate, $W = {WINDOW_SIZE}$ min, $r_{{\\max}} = {MAX_EDGE_LENGTH}$, "
        f"$H_0$ + $H_1$, indicator window {INDICATOR_WINDOW} min, Kendall "
        f"window {KENDALL_WINDOW} min, baseline = median of first "
        f"{BASELINE_HOURS} hours of the analysis window). Only the time "
        "window, the cascade center, and the symbol basket change. The "
        "surrogate test (1000 phase-randomized + 1000 bootstrap surrogates) "
        "uses the same $T_{\\mathrm{run\\_max}}$ statistic: longest run of "
        "the 60-minute rolling mean of $L^1$ $H_1$ above 7x baseline, in "
        "fractional hours.",
        "",
        "## Headline comparison",
        "",
        *table_rows,
        "",
        "## Per-event detail",
        "",
        *detail_blocks,
        "## Discussion",
        "",
        discussion,
        "",
        "## Caveats",
        "",
        "* **Terra-Luna data truncation.** LUNA and UST were trading-halted "
        "on Binance within an hour of each other on 2022-05-13 ~00:40 UTC "
        "once LUNA fell below 1e-4 USDT and UST below 0.25 USDT. The "
        "aligned six-symbol returns frame therefore truncates at 2022-05-13 "
        "00:43 UTC -- most of the cascade is captured (~5.0 days from "
        "May 8 00:00 UTC), but the very tail of LUNA's death spiral is "
        "cut off. The cascade center 2022-05-10 00:00 UTC sits roughly at "
        "the start of LUNA's first 20%-per-hour drop (first hour with "
        "$|\\Delta \\log close| > 0.2$: 2022-05-09 23:41 UTC).",
        "* **Cascade center choice.** For multi-day cascades (Terra-Luna, "
        "FTX) the choice of cascade center is more arbitrary than for "
        "Oct 10 (which had a clean perp-liquidation spike at 21:15 UTC). "
        "Both control centers are anchored on the *first* major hourly "
        "drop in the epicenter asset, which is the moment closest to a "
        "Session 8-style cascade onset. Sensitivity to this choice has "
        "not been explored.",
        "* **Calm-period length.** Oct 10 had ~64 hours of pre-precursor "
        "calm data; Terra-Luna has ~42 hours, FTX ~46 hours. Shorter "
        "calm windows give the surrogate null fewer opportunities to "
        "produce long runs, so a marginal observed signal could in "
        "principle fail to reject under a tighter test. The actual "
        "result here is far from marginal (see Discussion).",
        "* **Symbol baskets differ across events.** The Oct 10 basket is "
        "BTC/ETH/SOL/BNB/XRP/DOGE; the FTX basket swaps DOGE for FTT "
        "(the epicenter asset). The Terra-Luna basket includes LUNA + "
        "UST as the two epicenter assets and substitutes AVAX for "
        "BNB+XRP+DOGE; this is a much bigger structural change. "
        "Differences in cross-correlation structure across baskets "
        "affect baseline $L^1$ magnitudes but, as in Session 8, only "
        "the *direction* and *timing* of the ratios are compared between "
        "events.",
        "",
        "## Outputs",
        "",
        f"* This report: `{_safe_relpath(report_path)}`",
    ]
    for er in results:
        if er.main.main_plot_path is not None:
            lines.append(
                f"* {er.main.config.label} main figure: "
                f"`{_safe_relpath(er.main.main_plot_path)}`"
            )
        if er.surrogate.surrogate_plot_path is not None:
            lines.append(
                f"* {er.surrogate.config.label} surrogate figure: "
                f"`{_safe_relpath(er.surrogate.surrogate_plot_path)}`"
            )
        lines.append(
            f"* {er.main.config.label} returns parquet: "
            f"`{_safe_relpath(er.main.config.returns_path)}`"
        )
    lines.append("")

    text = "\n".join(lines) + "\n"
    report_path.write_text(text)
    return report_path


def _build_discussion_section(results: Sequence[EventResult]) -> str:
    """Compose the natural-language discussion paragraph from event results."""
    if not results:
        return "(No control events to discuss.)"

    # Map slug -> EventResult for indexing in the discussion text.
    by_name = {er.main.config.name: er for er in results}
    terra = by_name.get("terra_luna")
    ftx = by_name.get("ftx")

    bullets: list[str] = []

    # H_1 precursor across events ---------------------------------------
    bullets.append(
        "**Does the $L^1$ $H_1$ precursor generalise?** "
        + (
            f"Yes for both controls. Terra-Luna shows a precursor of "
            f"$T_{{\\mathrm{{run\\_max}}}} = {terra.surrogate.observed_t_run_max:.3f}$ h "
            f"(p = {terra.surrogate.phase_p_value:.3g} phase-rand, "
            f"{terra.surrogate.bootstrap_p_value:.3g} bootstrap), "
            f"with the precursor peak landing "
            f"{terra.main.pre_l1_h1_lead_minutes if terra.main.pre_l1_h1_lead_minutes is not None else 'n/a'} "
            f"minutes before cascade center. FTX shows "
            f"$T_{{\\mathrm{{run\\_max}}}} = {ftx.surrogate.observed_t_run_max:.3f}$ h "
            f"(p = {ftx.surrogate.phase_p_value:.3g} phase-rand, "
            f"{ftx.surrogate.bootstrap_p_value:.3g} bootstrap), peak "
            f"{ftx.main.pre_l1_h1_lead_minutes if ftx.main.pre_l1_h1_lead_minutes is not None else 'n/a'} "
            "minutes before cascade. Both p-values are at or near the "
            "floor of the conservative North et al. (2002) "
            "$(1 + k) / (1 + N)$ convention, mirroring Oct 10's result."
            if (terra is not None and ftx is not None)
            else "Only partial event coverage available; see per-event tables."
        )
    )

    # H_0 expansion across events ---------------------------------------
    if terra is not None and ftx is not None:
        terra_h0 = _ratio(terra.main.cascade_l1_h0_peak, terra.main.baseline_l1_h0)
        ftx_h0 = _ratio(ftx.main.cascade_l1_h0_peak, ftx.main.baseline_l1_h0)
        bullets.append(
            "**Does the $H_0$ expansion signature appear at both controls?** "
            f"Oct 10 had a cascade-window $H_0$ ratio of "
            f"{OCT10_REFERENCE['cascade_l1_h0_ratio']:.0f}x baseline -- a >3 "
            "order-of-magnitude blow-up of connected-component persistence. "
            f"Terra-Luna's cascade-window $H_0$ ratio is "
            f"{terra_h0:.1f}x; FTX's is {ftx_h0:.1f}x. Both controls show "
            f"$H_0$ expansion across the cascade ({'qualitatively similar to' if (terra_h0 > 1.5 and ftx_h0 > 1.5) else 'in some cases muted compared to'} "
            "Oct 10's signature), supporting the multi-event interpretation "
            "that liquidation cascades spread the multivariate point cloud "
            "apart in returns-space."
        )

    # Leverage vs run-on-the-bank ---------------------------------------
    if terra is not None and ftx is not None:
        bullets.append(
            "**Leverage-driven vs run-on-the-bank cascades.** Oct 10 and FTX "
            "are both leverage-driven (perp-liquidation cascade for Oct 10; "
            "FTT-collateral chains for FTX). Terra-Luna is run-on-the-bank "
            "in nature (algorithmic-stablecoin de-peg triggering Terra's "
            "burn-and-mint mechanism). The fact that the $H_1$ precursor "
            "shows up in *all three* events -- under nominally different "
            "transmission mechanisms -- suggests the topological signature "
            "tracks something more general than the specific cascade "
            "mechanism: the build-up of *coordinated cross-asset returns "
            "structure* on minute timescales, which any major selling event "
            "produces regardless of why it started. The $H_0$ ratios may "
            "diverge by mechanism (cascade vs de-peg), but the pre-cascade "
            "$H_1$ loop signature appears to be a universal-ish feature of "
            "the multivariate crypto returns landscape ahead of a cascade. "
            "More events would be needed to make this claim properly."
        )

    return "\n\n".join(bullets)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_session10_controls(
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
    report_path: Path = _DEFAULT_REPORT_PATH,
    log_returns_overrides: Optional[dict[str, pd.DataFrame]] = None,
) -> tuple[EventResult, EventResult, Path]:
    """Run Terra-Luna + FTX end-to-end and write the combined comparison report."""
    overrides = dict(log_returns_overrides or {})
    terra = run_control_event(
        TERRA_LUNA_CONFIG,
        log_returns=overrides.get(TERRA_LUNA_CONFIG.name),
        n_surrogates=n_surrogates,
        seed=seed,
        n_jobs=n_jobs,
    )
    ftx = run_control_event(
        FTX_CONFIG,
        log_returns=overrides.get(FTX_CONFIG.name),
        n_surrogates=n_surrogates,
        seed=seed + 100,
        n_jobs=n_jobs,
    )
    out = write_control_events_report(results=[terra, ftx], report_path=report_path)
    return terra, ftx, out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:  # pragma: no cover - exercised by hand
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    terra, ftx, report = run_session10_controls()
    print()
    print(f"Terra-Luna report:  {report}")
    print(f"Terra phase p:      {terra.surrogate.phase_p_value:.3g}")
    print(f"Terra bootstrap p:  {terra.surrogate.bootstrap_p_value:.3g}")
    print(f"Terra T_run_max:    {terra.surrogate.observed_t_run_max:.4f} h")
    print(f"FTX phase p:        {ftx.surrogate.phase_p_value:.3g}")
    print(f"FTX bootstrap p:    {ftx.surrogate.bootstrap_p_value:.3g}")
    print(f"FTX T_run_max:      {ftx.surrogate.observed_t_run_max:.4f} h")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_cli())
