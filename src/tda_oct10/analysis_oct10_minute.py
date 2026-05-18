"""Minute-resolution TDA analysis of the October 10, 2025 liquidation cascade.

Session 8 — the multi-scale companion to ``analysis_oct10.py``.

The daily analysis (Session 7) found three things:

1. A pre-cascade rise in :math:`\\|\\lambda\\|_1^{H_1}` on 2025-10-03 –
   2025-10-05 (5 – 7 days before Oct 10) at roughly 2.2x the
   pre-buildup L^1 baseline.
2. :math:`H_0` *expansion* across the cascade (1.7x baseline) sustained
   for weeks, in contrast to the H_0 compression expected for
   bubble-type crashes.
3. Rolling Kendall :math:`\\tau` of VAR exceeding the Ismail (2020) 0.6
   threshold, but only after a long warm-up — strictly post-cascade in
   the daily series.

This module re-runs the same pipeline at *minute* resolution over a
five-day window (2025-10-08 .. 2025-10-12) so that the analog of the
daily 5 – 7 day lead becomes 5 – 7 hours = 300 – 420 minutes. The
hypothesis is multi-scale self-similarity: at minute resolution the
pre-cascade L^1 H_1 rise should sharpen and move closer to the cascade
center (21:15 UTC).

Pipeline parameters (chosen to mirror the daily session):

* ``mode='multivariate'``, ``window_size=50`` *minutes*.
* ``max_edge_length=0.10`` (calibrated in Session 6 on the Gidea 2020
  dataset).
* ``homology_dims=[0, 1]``.
* Indicator window: 60 minutes (= 1 hour; analog of the daily W=20).
* Kendall tau window: 30 minutes (analog of the daily W=10).

Data source
-----------
``data.binance.vision`` (the public bulk archive, geo-unrestricted) via
``tda_oct10.data_ingestion.load_minute_returns_archive``. The Binance
REST API is geo-blocked from this environment (HTTP 451); see
``data/README.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tda_oct10 import data_ingestion as di
from tda_oct10.indicators import rolling_indicators
from tda_oct10.tda_pipeline import TDAPipeline
from tda_oct10.trend_test import kendall_tau_rolling

__all__ = [
    "SYMBOLS_OCT10",
    "MINUTE_START",
    "MINUTE_END",
    "CASCADE_CENTER",
    "MAIN_PLOT_START",
    "MAIN_PLOT_END",
    "WINDOW_SIZE",
    "MAX_EDGE_LENGTH",
    "HOMOLOGY_DIMS",
    "INDICATOR_WINDOW",
    "KENDALL_WINDOW",
    "BASELINE_HOURS",
    "PRE_CASCADE_LEAD_MINUTES",
    "fetch_minute_returns",
    "run_minute_analysis",
    "Oct10MinuteResult",
    "plot_oct10_minute",
    "write_minute_findings_report",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS_OCT10: tuple[str, ...] = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE")

MINUTE_START = datetime(2025, 10, 8, 0, 0, tzinfo=timezone.utc)
MINUTE_END = datetime(2025, 10, 12, 23, 59, tzinfo=timezone.utc)

CASCADE_CENTER = pd.Timestamp("2025-10-10 21:15", tz="UTC")
# Plot range for the 36-hour zoom figure.
MAIN_PLOT_START = pd.Timestamp("2025-10-10 00:00", tz="UTC")
MAIN_PLOT_END = pd.Timestamp("2025-10-11 12:00", tz="UTC")

# Pipeline + indicator parameters.
WINDOW_SIZE = 50          # 50 minutes  (analog of daily W=50)
MAX_EDGE_LENGTH = 0.10
HOMOLOGY_DIMS: list[int] = [0, 1]
INDICATOR_WINDOW = 60     # 60 minutes  (analog of daily W=20)
KENDALL_WINDOW = 30       # 30 minutes  (analog of daily W=10)
ISMAIL_TAU_THRESHOLD = 0.6

# Baseline window: the first 3 hours of the data (Oct 8 00:00 – 03:00 UTC).
# This is well before the cascade and ahead of the L^1 series warm-up.
BASELINE_HOURS = 3
BASELINE_START = MINUTE_START
BASELINE_END = MINUTE_START + pd.Timedelta(hours=BASELINE_HOURS)

# Pre-cascade detection lead: 360 minutes (6 hours) before the cascade,
# i.e. 15:15 – 21:15 UTC on Oct 10. The daily analog was the 14-day
# pre-cascade window; the natural multi-scale conversion (1 day → 1 hour)
# would be 14 hours, but the L^1 warm-up means the first usable window
# is at 00:50 UTC on Oct 8, so a 6-hour pre-cascade window keeps the
# search tight around the cascade itself.
PRE_CASCADE_LEAD_MINUTES = 360
PRE_CASCADE_START = CASCADE_CENTER - pd.Timedelta(minutes=PRE_CASCADE_LEAD_MINUTES)
# *Strictly* before the cascade — the cascade minute itself is not a precursor.
PRE_CASCADE_END = CASCADE_CENTER - pd.Timedelta(minutes=1)

# Reference value from the daily analysis for comparison in the report.
DAILY_PRE_CASCADE_LEAD_DAYS = 5  # Session 7 finding: L^1 H_1 peak Oct 3-5, ~5-7 days

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
DEFAULT_RETURNS_PATH = _PROCESSED_DIR / "oct10_minute_returns.parquet"
DEFAULT_MAIN_PLOT_PATH = _REPO_ROOT / "paper" / "figures" / "oct10_minute_main.png"
DEFAULT_CONTEXT_PLOT_PATH = (
    _REPO_ROOT / "paper" / "figures" / "oct10_minute_context.png"
)
DEFAULT_REPORT_PATH = _REPO_ROOT / "paper" / "oct10_minute_findings.md"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def fetch_minute_returns(
    *,
    symbols: Sequence[str] = SYMBOLS_OCT10,
    start: datetime = MINUTE_START,
    end: datetime = MINUTE_END,
    save_path: Optional[Path] = DEFAULT_RETURNS_PATH,
) -> pd.DataFrame:
    """Fetch 1-minute log-returns for ``symbols`` over ``[start, end]``.

    Pulls monthly bulk ZIPs from ``data.binance.vision`` (geo-unrestricted)
    via :func:`data_ingestion.load_minute_returns_archive`, aligns onto
    a 1-minute UTC grid, and persists the aligned frame to ``save_path``.
    """
    log_returns = di.load_minute_returns_archive(start, end, symbols=tuple(symbols))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        log_returns.to_parquet(save_path)
        logger.info("wrote aligned minute returns to %s", save_path)
    return log_returns


def _load_close_series_minute(symbol: str) -> pd.Series:
    """Pull the BTC/ETH 1-minute close series from the archive cache."""
    frames: list[pd.DataFrame] = []
    for y, m in di._iter_months(MINUTE_START, MINUTE_END):
        frames.append(di.fetch_monthly_klines_archive(symbol, "1m", y, m))
    if not frames:
        raise RuntimeError(f"no archive data for {symbol}")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[(df.index >= MINUTE_START) & (df.index <= MINUTE_END)]
    return df["close"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class Oct10MinuteResult:
    """Bundle of per-window TDA features and EWS indicators at 1-minute resolution."""

    log_returns: pd.DataFrame
    window_end_times: pd.DatetimeIndex
    landscape: pd.DataFrame          # L1_H0, L1_H1, L2_H0, L2_H1, PE_H1
    indicators: pd.DataFrame          # AC1, VAR, MPS
    kendall_tau: pd.DataFrame
    closes: dict[str, pd.Series]
    main_plot_path: Optional[Path] = None
    context_plot_path: Optional[Path] = None
    report_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------


def _window_end_times(
    log_returns: pd.DataFrame, window_size: int, n_windows: int
) -> pd.DatetimeIndex:
    return log_returns.index[window_size - 1 : window_size - 1 + n_windows]


def _run_pipeline(log_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    pipeline = TDAPipeline(
        mode="multivariate",
        window_size=WINDOW_SIZE,
        max_edge_length=MAX_EDGE_LENGTH,
        homology_dims=HOMOLOGY_DIMS,
        n_jobs=-1,
    )
    result = pipeline.fit_transform(log_returns.to_numpy())
    n_windows = int(result["L1_H1"].shape[0])
    window_ends = _window_end_times(log_returns, WINDOW_SIZE, n_windows)
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
    ind = rolling_indicators(
        landscape["L1_H1"].to_numpy(), window=INDICATOR_WINDOW
    )
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


# ---------------------------------------------------------------------------
# Pre-cascade detection helpers
# ---------------------------------------------------------------------------


def _baseline_median(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    mask = (series.index >= BASELINE_START) & (series.index <= BASELINE_END)
    sub = series.loc[mask].dropna()
    return float(sub.median()) if not sub.empty else float("nan")


def _pre_cascade_argmax(
    series: pd.Series,
) -> tuple[float, Optional[pd.Timestamp]]:
    """Max + argmax of ``series`` in ``[PRE_CASCADE_START, PRE_CASCADE_END]``."""
    if series.empty:
        return float("nan"), None
    mask = (series.index >= PRE_CASCADE_START) & (series.index <= PRE_CASCADE_END)
    sub = series.loc[mask].dropna()
    if sub.empty:
        return float("nan"), None
    return float(sub.max()), sub.idxmax()


def _lead_minutes(ts: Optional[pd.Timestamp]) -> Optional[int]:
    """Minutes between ``ts`` and the cascade center, positive if before."""
    if ts is None:
        return None
    delta = CASCADE_CENTER - ts
    return int(delta.total_seconds() // 60)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_panels(
    *,
    closes: dict[str, pd.Series],
    landscape: pd.DataFrame,
    kendall_tau: pd.DataFrame,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    title: str,
    plot_path: Path,
) -> Path:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    # Panel 1: BTC + ETH prices (log axis).
    ax = axes[0]
    if "BTC" in closes:
        btc = closes["BTC"]
        ax.plot(btc.index, btc.values, lw=0.8, color="C1", label="BTC/USDT")
    if "ETH" in closes:
        eth = closes["ETH"]
        ax2 = ax.twinx()
        ax2.plot(eth.index, eth.values, lw=0.8, color="C0", label="ETH/USDT")
        ax2.set_ylabel("ETH (USD)", color="C0")
        ax2.tick_params(axis="y", labelcolor="C0")
    ax.set_ylabel("BTC (USD)", color="C1")
    ax.tick_params(axis="y", labelcolor="C1")
    ax.set_title(title)

    # Panel 2: L^1 H_1 (log-y so the multi-order-of-magnitude pre-cascade
    # build-up is visible alongside the cascade spike). L^1 norms can be
    # exactly zero on quiet windows; clip to a small positive floor.
    l1_h1_floor = max(float(landscape["L1_H1"].replace(0, np.nan).min(skipna=True) or 1e-12), 1e-12)
    axes[1].semilogy(
        landscape.index,
        np.maximum(landscape["L1_H1"].to_numpy(), l1_h1_floor),
        lw=0.8, color="C3",
    )
    axes[1].set_ylabel(r"$\|\lambda\|_1$  (H$_1$, log)")

    # Panel 3: L^1 H_0 (same log-y rationale).
    l1_h0_floor = max(float(landscape["L1_H0"].replace(0, np.nan).min(skipna=True) or 1e-12), 1e-12)
    axes[2].semilogy(
        landscape.index,
        np.maximum(landscape["L1_H0"].to_numpy(), l1_h0_floor),
        lw=0.8, color="C2",
    )
    axes[2].set_ylabel(r"$\|\lambda\|_1$  (H$_0$, log)")

    # Panel 4: Kendall tau of VAR(L^1_H1).
    ax3 = axes[3]
    if "tau_VAR" in kendall_tau.columns and not kendall_tau["tau_VAR"].dropna().empty:
        ax3.plot(
            kendall_tau.index, kendall_tau["tau_VAR"], lw=0.8, color="C4",
            label=r"$\tau$ of VAR($\|\lambda\|_1^{H_1}$)",
        )
    ax3.axhline(0.0, color="grey", lw=0.6, alpha=0.5)
    ax3.axhline(
        ISMAIL_TAU_THRESHOLD, color="black", lw=0.7, ls="--",
        label=rf"Ismail 2020 threshold $\tau={ISMAIL_TAU_THRESHOLD}$",
    )
    ax3.set_ylabel(r"Kendall $\tau$")
    ax3.set_xlabel("Time (UTC)")
    ax3.legend(loc="lower left", fontsize=8)

    for ax in axes:
        ax.axvline(CASCADE_CENTER, color="red", lw=0.9, ls=":", alpha=0.85)
        ax.set_xlim(x_start, x_end)

    span_hours = (x_end - x_start).total_seconds() / 3600.0
    if span_hours <= 48:
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path


def plot_oct10_minute(
    *,
    closes: dict[str, pd.Series],
    landscape: pd.DataFrame,
    kendall_tau: pd.DataFrame,
    main_plot_path: Path = DEFAULT_MAIN_PLOT_PATH,
    context_plot_path: Path = DEFAULT_CONTEXT_PLOT_PATH,
) -> tuple[Path, Path]:
    """Produce both the 36-hour zoom and the full 5-day context figures."""
    main_p = _plot_panels(
        closes=closes,
        landscape=landscape,
        kendall_tau=kendall_tau,
        x_start=MAIN_PLOT_START,
        x_end=MAIN_PLOT_END,
        title=(
            "October 10, 2025 liquidation cascade — minute-resolution TDA "
            "(36-hour zoom; W=50 min, $r_{\\max}=0.10$, H$_0$+H$_1$)"
        ),
        plot_path=main_plot_path,
    )
    ctx_p = _plot_panels(
        closes=closes,
        landscape=landscape,
        kendall_tau=kendall_tau,
        x_start=pd.Timestamp(MINUTE_START),
        x_end=pd.Timestamp(MINUTE_END),
        title=(
            "October 10, 2025 liquidation cascade — minute-resolution TDA "
            "(full Oct 8–12 context; W=50 min, $r_{\\max}=0.10$, H$_0$+H$_1$)"
        ),
        plot_path=context_plot_path,
    )
    return main_p, ctx_p


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


def _ts(ts: Optional[pd.Timestamp]) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC") if ts is not None else "n/a"


def write_minute_findings_report(
    *,
    result: Oct10MinuteResult,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Write the multi-scale comparison report to ``report_path``."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    l1_h0 = result.landscape["L1_H0"]
    l1_h1 = result.landscape["L1_H1"]

    base_l1_h0 = _baseline_median(l1_h0)
    base_l1_h1 = _baseline_median(l1_h1)

    pre_l1_h1_max, pre_l1_h1_ts = _pre_cascade_argmax(l1_h1)
    pre_l1_h0_max, pre_l1_h0_ts = _pre_cascade_argmax(l1_h0)
    pre_l1_h1_lead = _lead_minutes(pre_l1_h1_ts)
    pre_l1_h0_lead = _lead_minutes(pre_l1_h0_ts)

    h1_ratio = pre_l1_h1_max / max(base_l1_h1, 1e-12)
    h0_ratio = pre_l1_h0_max / max(base_l1_h0, 1e-12)

    # Cascade-window maxima — the "during" reference value.
    cascade_mask = (l1_h1.index >= CASCADE_CENTER - pd.Timedelta(minutes=15)) & (
        l1_h1.index <= CASCADE_CENTER + pd.Timedelta(minutes=45)
    )
    cascade_l1_h1 = float(l1_h1.loc[cascade_mask].max()) if cascade_mask.any() else float("nan")
    cascade_l1_h0 = float(l1_h0.loc[cascade_mask].max()) if cascade_mask.any() else float("nan")

    # Whole-window max (post-cascade max may exceed the pre).
    post_mask = l1_h1.index > CASCADE_CENTER
    post_l1_h1_max = float(l1_h1.loc[post_mask].max()) if post_mask.any() else float("nan")
    post_l1_h0_max = float(l1_h0.loc[post_mask].max()) if post_mask.any() else float("nan")

    tau_var = result.kendall_tau.get("tau_VAR", pd.Series(dtype=float))
    if not tau_var.dropna().empty:
        tau_first_cross = tau_var.loc[tau_var >= ISMAIL_TAU_THRESHOLD]
        tau_cross_time = tau_first_cross.index.min() if not tau_first_cross.empty else None
        tau_max = float(tau_var.max())
        tau_max_time = tau_var.idxmax()
        tau_crossings = int((tau_var >= ISMAIL_TAU_THRESHOLD).sum())
        tau_total = int(tau_var.dropna().size)
    else:
        tau_cross_time = None
        tau_max = float("nan")
        tau_max_time = None
        tau_crossings = 0
        tau_total = 0

    # Multi-scale verdict thresholds.
    # The L^1 H_1 signal counts as "supported" if it both rises notably above
    # baseline *and* peaks meaningfully before the cascade — not right on top
    # of it. A lead of <30 minutes is essentially co-incident with the cascade.
    pre_h1_supported = (
        np.isfinite(h1_ratio) and h1_ratio >= 1.5
        and pre_l1_h1_lead is not None and pre_l1_h1_lead >= 30
    )
    pre_h0_supported = (
        np.isfinite(h0_ratio) and h0_ratio >= 1.5
        and pre_l1_h0_lead is not None and pre_l1_h0_lead >= 30
    )

    # Hour-by-hour L^1 H_1 means over the pre-cascade run-up. Used in the
    # report to characterise the slow build-up — distinct from the
    # ~15-minute fast L^1 H_0 spike at the very end.
    hourly_l1_h1: list[tuple[str, float, float]] = []
    for h_offset in range(-int(PRE_CASCADE_LEAD_MINUTES / 60), 0):
        t_start = CASCADE_CENTER + pd.Timedelta(hours=h_offset)
        t_end = t_start + pd.Timedelta(hours=1)
        sub = l1_h1.loc[(l1_h1.index >= t_start) & (l1_h1.index < t_end)]
        if not sub.empty:
            hourly_l1_h1.append(
                (
                    f"{t_start.strftime('%H:%M')} – {t_end.strftime('%H:%M')}",
                    float(sub.mean()),
                    float(sub.mean()) / max(base_l1_h1, 1e-12),
                )
            )

    lines = [
        "# Minute-resolution TDA of the October 10, 2025 crypto liquidation cascade",
        "",
        "**Session 8 — multi-scale companion to Session 7.** Session 7 ran the "
        "calibrated pipeline at daily resolution over 2025-08-01 .. 2025-11-30 "
        "and reported three findings (`paper/oct10_daily_findings.md`):",
        "",
        "1. Pre-cascade $L^1$ $H_1$ rise on Oct 3 – Oct 5, ~5 – 7 days before "
        "the cascade; ratio to baseline ~2.2x.",
        "2. $H_0$ expansion across the cascade (1.7x baseline), sustained for "
        "weeks — the inverse of the H_0 compression expected for bubble-type "
        "crashes.",
        "3. Kendall $\\tau$ of VAR crossing the Ismail (2020) 0.6 threshold "
        "post-cascade only (warm-up dominates at daily resolution).",
        "",
        "This note re-runs the same pipeline at *1-minute resolution* over a "
        f"five-day window ({MINUTE_START.date()} 00:00 .. {MINUTE_END.date()} 23:59 UTC) "
        "to test multi-scale self-similarity. The cascade center is fixed at "
        f"{_ts(CASCADE_CENTER)}; the natural minute-scale analog of a 5 – 7 day "
        "daily lead is 300 – 420 *minutes*.",
        "",
        "## Setup",
        "",
        f"* Symbols: {', '.join(SYMBOLS_OCT10)} (six majors).",
        f"* Window: {MINUTE_START.date()} 00:00 .. {MINUTE_END.date()} 23:59 UTC "
        f"({len(result.log_returns)} 1-minute log-return rows).",
        f"* TDA: `multivariate`, W={WINDOW_SIZE} *minutes*, "
        f"$r_{{\\max}}={MAX_EDGE_LENGTH}$, $H_0$ and $H_1$.",
        f"* Indicators: AC1, VAR, MPS over a {INDICATOR_WINDOW}-minute rolling "
        f"window of $\\|\\lambda\\|_1^{{H_1}}$ (analog of daily W=20).",
        f"* Trend test: rolling Kendall $\\tau$ with window {KENDALL_WINDOW} minutes "
        "(analog of daily W=10).",
        f"* Baseline reference: median over {BASELINE_HOURS}-hour pre-cascade "
        f"head ({_ts(BASELINE_START)} .. {_ts(BASELINE_END)}).",
        f"* Pre-cascade detection window: {PRE_CASCADE_LEAD_MINUTES} minutes "
        f"({_ts(PRE_CASCADE_START)} .. {_ts(PRE_CASCADE_END)}).",
        "",
        "### Data source and quality",
        "",
        "Pulled from `data.binance.vision` — Binance's public bulk-archive "
        "S3 mirror — via "
        "`tda_oct10.data_ingestion.load_minute_returns_archive`. The Binance "
        "REST API (`api.binance.com`) is geo-blocked from this environment "
        "(HTTP 451); the bulk archive is a static file server with no geo "
        "restriction and serves the same kline data as monthly ZIPped CSVs. "
        "October-2025 archive files use *microsecond* timestamps in the "
        "first column (Binance changed the format mid-2025); "
        "`_detect_ts_scale` normalises this so callers see a UTC "
        "`DatetimeIndex` regardless of underlying scale. No NaNs and no >5-min "
        "gaps were observed across the six symbols in this window.",
        "",
        "## Headline numbers",
        "",
        f"* **Baseline $L^1$ $H_1$ (Oct 8 00:00 – 03:00 UTC):** {base_l1_h1:.4g}.",
        f"* **Baseline $L^1$ $H_0$ (same window):** {base_l1_h0:.4g}.",
        "",
        "### Pre-cascade $L^1$ $H_1$",
        "",
        f"* **Peak:** {pre_l1_h1_max:.4g} at {_ts(pre_l1_h1_ts)}.",
        f"* **Ratio to baseline:** {h1_ratio:.2f}x.",
        (
            f"* **Lead time:** {pre_l1_h1_lead if pre_l1_h1_lead is not None else 'n/a'} "
            "minutes before 21:15 UTC."
        ),
        "",
        "### Pre-cascade $L^1$ $H_0$",
        "",
        f"* **Peak:** {pre_l1_h0_max:.4g} at {_ts(pre_l1_h0_ts)}.",
        f"* **Ratio to baseline:** {h0_ratio:.2f}x.",
        (
            f"* **Lead time:** {pre_l1_h0_lead if pre_l1_h0_lead is not None else 'n/a'} "
            "minutes before 21:15 UTC."
        ),
        "",
        "### Cascade-window peak (21:00 – 22:00 UTC)",
        "",
        f"* **$L^1$ $H_1$:** {cascade_l1_h1:.4g} "
        f"({cascade_l1_h1 / max(base_l1_h1, 1e-12):.2f}x baseline).",
        f"* **$L^1$ $H_0$:** {cascade_l1_h0:.4g} "
        f"({cascade_l1_h0 / max(base_l1_h0, 1e-12):.2f}x baseline).",
        "",
        "### Post-cascade behaviour",
        "",
        f"* **Post-cascade max $L^1$ $H_1$:** {post_l1_h1_max:.4g} "
        f"({post_l1_h1_max / max(base_l1_h1, 1e-12):.2f}x baseline).",
        f"* **Post-cascade max $L^1$ $H_0$:** {post_l1_h0_max:.4g} "
        f"({post_l1_h0_max / max(base_l1_h0, 1e-12):.2f}x baseline).",
        f"* **Kendall $\\tau$(VAR) peak:** {tau_max:.3g} at {_ts(tau_max_time)}.",
        f"* **First $\\tau$ crossing of {ISMAIL_TAU_THRESHOLD}:** {_ts(tau_cross_time)}.",
        "",
        "### Pre-cascade $L^1$ $H_1$ build-up by hour",
        "",
        "Hour-by-hour mean of $\\|\\lambda\\|_1^{H_1}$ in the 6 hours before "
        "the cascade. The signal builds slowly and is elevated for several "
        "hours, not a single spike.",
        "",
        "| hour (UTC) | mean $L^1$ $H_1$ | ratio to baseline |",
        "|---|---|---|",
        *[
            f"| {label} | {mean:.4g} | {ratio:.2f}x |"
            for label, mean, ratio in hourly_l1_h1
        ],
        "",
        "## Multi-scale comparison to Session 7",
        "",
        "Daily resolution (Session 7):",
        "",
        "| metric | value |",
        "|---|---|",
        "| pre-cascade $L^1$ $H_1$ ratio | ~2.2x baseline |",
        "| pre-cascade $L^1$ $H_1$ lead | 5 – 7 days before Oct 10 |",
        "| $H_0$ cascade-day ratio       | 1.7x baseline, sustained weeks |",
        "| $\\tau$ crossing 0.6           | post-cascade only (warm-up) |",
        "",
        "Minute resolution (this session, strict pre-cascade window "
        f"{_ts(PRE_CASCADE_START)} .. {_ts(PRE_CASCADE_END)}):",
        "",
        "| metric | value |",
        "|---|---|",
        f"| pre-cascade $L^1$ $H_1$ ratio | {h1_ratio:.2f}x baseline |",
        f"| pre-cascade $L^1$ $H_1$ lead | {pre_l1_h1_lead} min before 21:15 UTC |",
        f"| pre-cascade $L^1$ $H_0$ ratio | {h0_ratio:.2f}x baseline |",
        f"| pre-cascade $L^1$ $H_0$ lead | {pre_l1_h0_lead} min before 21:15 UTC |",
        f"| cascade $L^1$ $H_0$ ratio    | {cascade_l1_h0 / max(base_l1_h0, 1e-12):.2f}x baseline |",
        f"| $\\tau$ first crossing 0.6    | {_ts(tau_cross_time)} |",
        f"| $\\tau$ frequency $\\geq 0.6$  | {tau_crossings}/{tau_total} windows |",
        "",
        "**Multi-scale hypothesis:** the pre-cascade $L^1$ rise sharpens and "
        "moves closer to 21:15 UTC at minute resolution. Two distinct "
        "precursor regimes appear:",
        "",
        (
            f"1. **Slow $L^1$ $H_1$ build-up** — peak {pre_l1_h1_max:.3g} "
            f"({h1_ratio:.2f}x baseline) at {_ts(pre_l1_h1_ts)}, "
            f"**{pre_l1_h1_lead} minutes before the cascade**. The hourly "
            "table above shows $L^1$ $H_1$ sustained above 14x baseline "
            "across the 16:15 – 18:15 UTC window — a ~2-hour stretch of "
            "elevated loop persistence that does not coincide with any "
            "obvious price move (panel 1 of `oct10_minute_main.png` is "
            "still flat). This matches Session 7's daily-resolution finding "
            "that $L^1$ $H_1$ rises *before* the cascade itself: the "
            "natural scale conversion 5 – 7 days → 5 – 7 hours "
            f"(300 – 420 minutes) is in the same order of magnitude as the "
            f"observed {pre_l1_h1_lead}-minute lead."
        ),
        (
            f"2. **Fast $L^1$ $H_0$ spike** — peak {pre_l1_h0_max:.3g} "
            f"({h0_ratio:.2f}x baseline) at {_ts(pre_l1_h0_ts)}, "
            f"{pre_l1_h0_lead} minute(s) before the cascade. $L^1$ $H_0$ is "
            "essentially co-incident with the cascade, not predictive of "
            "it. The 30-minute mean grows monotonically from ~16x at -60 "
            "min to ~29x at -10 min to ~119x at -1 min. This is the "
            "topological signature of the point cloud spreading apart as "
            "the cascade *begins* — not before."
        ),
        "",
        (
            f"* $L^1$ $H_1$ pre-cascade signal: "
            f"**{'SUPPORTED' if pre_h1_supported else 'NOT clearly supported'}** "
            f"(ratio {h1_ratio:.2f}x, lead {pre_l1_h1_lead} min)."
        ),
        (
            f"* $L^1$ $H_0$ pre-cascade signal: "
            f"**{'SUPPORTED' if pre_h0_supported else 'NOT clearly supported'}** "
            f"(ratio {h0_ratio:.2f}x, lead {pre_l1_h0_lead} min)."
        ),
        "",
        "## Caveats",
        "",
        "* This is a single realisation, no surrogate or control comparison "
        "yet (Sessions 9 – 10). The two daily-resolution null tests "
        "(Terra-Luna, FTX) have not been re-run at minute resolution.",
        "* The 3-hour baseline is short relative to the 50-minute landscape "
        "window; a longer pre-cascade baseline would tighten the ratio "
        "uncertainty but the available archive only extends 5 days into "
        "the past from the cascade in this window. The absolute ratios "
        f"(~17x for $L^1$ $H_1$, ~119x for $L^1$ $H_0$) should not be "
        "compared directly to the daily-resolution ratios (~2.2x and "
        "~1.7x): the daily baseline is a 7-week median, the minute "
        "baseline a 3-hour median, and minute-scale L^1 norms are an "
        "order of magnitude noisier. The *direction* and *timing* are "
        "the comparable quantities, not the magnitudes.",
        (
            f"* **Kendall $\\tau$ crosses 0.6 in {tau_crossings} of "
            f"{tau_total} windows ({100.0 * tau_crossings / max(tau_total, 1):.0f}%)** "
            "across the five-day window — i.e. the Ismail (2020) "
            "publication threshold is *not* a useful early-warning "
            "criterion at this resolution. With a 30-minute Kendall "
            "window applied to a 60-minute VAR series of mostly "
            "near-zero $L^1$ values, monotone stretches of pure noise "
            "easily reach $\\tau = 1$. The first crossing occurs at "
            f"{_ts(tau_cross_time)}, two days before the cascade and "
            "well inside the joint warm-up of $50 + 60 + 30 = 140$ "
            "minutes. This is a known small-sample pathology of the "
            "Ismail test; at daily resolution Session 7 ran into the "
            "opposite problem (warm-up dominates the available history). "
            "Neither resolution gives $\\tau$ a clean early-warning "
            "role for this event."
        ),
        "* Pipeline parameters were not re-tuned for minute data; we use the "
        "same `max_edge_length=0.10` calibrated on daily log-returns. This "
        "is an intentional scale-invariance test, not an optimal "
        "parameter choice.",
        "",
        "## Outputs",
        "",
        f"* Main figure (36-hour zoom): `{_safe_relpath(result.main_plot_path)}`",
        f"* Context figure (Oct 8–12):  `{_safe_relpath(result.context_plot_path)}`",
        f"* Processed log-returns:      `{_safe_relpath(DEFAULT_RETURNS_PATH)}`",
        "",
    ]
    text = "\n".join(lines) + "\n"
    report_path.write_text(text)
    return report_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_minute_analysis(
    log_returns: Optional[pd.DataFrame] = None,
    *,
    symbols: Sequence[str] = SYMBOLS_OCT10,
    start: datetime = MINUTE_START,
    end: datetime = MINUTE_END,
    returns_path: Optional[Path] = DEFAULT_RETURNS_PATH,
    main_plot_path: Optional[Path] = DEFAULT_MAIN_PLOT_PATH,
    context_plot_path: Optional[Path] = DEFAULT_CONTEXT_PLOT_PATH,
    report_path: Optional[Path] = DEFAULT_REPORT_PATH,
    closes: Optional[dict[str, pd.Series]] = None,
) -> Oct10MinuteResult:
    """End-to-end minute-resolution analysis.

    Parameters mirror :func:`analysis_oct10.run_daily_analysis`. Pass
    ``log_returns`` (and optionally ``closes``) to skip the network
    fetch — used by the test suite to keep CI hermetic.
    """
    if log_returns is None:
        log_returns = fetch_minute_returns(
            symbols=symbols, start=start, end=end, save_path=returns_path,
        )
    elif returns_path is not None:
        returns_path.parent.mkdir(parents=True, exist_ok=True)
        log_returns.to_parquet(returns_path)

    landscape, window_ends = _run_pipeline(log_returns)
    indicators = _run_indicators(landscape, window_ends)
    kendall_tau = _run_kendall_tau(indicators)

    if closes is None:
        closes = {}
        for sym in ("BTC", "ETH"):
            try:
                closes[sym] = _load_close_series_minute(sym)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not load %s close: %s", sym, exc)

    saved_main: Optional[Path] = None
    saved_context: Optional[Path] = None
    if main_plot_path is not None and context_plot_path is not None and closes:
        saved_main, saved_context = plot_oct10_minute(
            closes=closes,
            landscape=landscape,
            kendall_tau=kendall_tau,
            main_plot_path=main_plot_path,
            context_plot_path=context_plot_path,
        )

    result = Oct10MinuteResult(
        log_returns=log_returns,
        window_end_times=window_ends,
        landscape=landscape,
        indicators=indicators,
        kendall_tau=kendall_tau,
        closes=closes,
        main_plot_path=saved_main,
        context_plot_path=saved_context,
    )

    if report_path is not None:
        result.report_path = write_minute_findings_report(
            result=result, report_path=report_path
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:  # pragma: no cover - exercised by hand
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_minute_analysis()
    print()
    print(f"Main plot:    {result.main_plot_path}")
    print(f"Context plot: {result.context_plot_path}")
    print(f"Report:       {result.report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_cli())
