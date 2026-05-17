"""Validation.

End-to-end validation utilities: synthetic-system sanity checks (e.g. Lorenz,
noisy circles), parameter sensitivity sweeps over Takens embedding dimension
and delay, and reproducibility checks for the full pipeline.

Default ``GIDEA_MAX_EDGE_LENGTH = 0.10`` calibrated by parameter sweep on
BTC/ETH/LTC/XRP daily log-returns, 2016-01-02 to 2018-01-07 (CryptoCompare
source). See ``paper/figures/gidea2020_epsilon_sweep.png``.

This module also reproduces published TDA-on-crypto results as out-of-sample
sanity checks before applying the pipeline to the Oct 10, 2025 cascade.
``replicate_gidea2020_fig7`` reproduces Figure 7c of

    Gidea, Goldsmith, Katz, Roldan & Shmalo (2020),
    "Topological recognition of critical transitions in time series of
    cryptocurrencies", Physica A 548.

The paper studies BTC, ETH, LTC, XRP daily closes from 2016-01-01 to
2018-01-07. The project's primary ``data_ingestion`` module is Binance-only
and cannot supply this range (Binance launched 2017-07; BTCUSDT klines start
2017-08-17), and its hard-coded symbol universe lacks LTC. So this module
fetches daily closes from CryptoCompare's public ``histoday`` endpoint
(no auth required) and applies the same align-then-log-returns recipe as
``data_ingestion.load_aligned_returns``: build the expected daily grid,
reindex every symbol onto it, require coverage across all symbols, then
take ``log(close).diff()``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from tda_oct10.tda_pipeline import TDAPipeline

__all__ = [
    "fetch_daily_close",
    "align_log_returns",
    "replicate_gidea2020_fig7",
    "Gidea2020Result",
    "sweep_max_edge_length",
    "EpsSweepResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRYPTOCOMPARE_HISTODAY = "https://min-api.cryptocompare.com/data/v2/histoday"

GIDEA_SYMBOLS: tuple[str, ...] = ("BTC", "ETH", "LTC", "XRP")
GIDEA_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
GIDEA_END = datetime(2018, 1, 7, tzinfo=timezone.utc)

# Paper parameters (Section 4, Figure 7). ``GIDEA_MAX_EDGE_LENGTH`` is set
# to 0.10 — the smallest value in ``sweep_max_edge_length(...)`` that
# reproduces Fig. 7c qualitatively on raw daily log-returns. The paper's
# implied 0.05 only works after per-window normalisation, which we
# deliberately do not apply (downstream Oct-10 analysis is on raw
# log-returns at the same scale).
GIDEA_WINDOW_SIZE = 50
GIDEA_MAX_EDGE_LENGTH = 0.10
GIDEA_HOMOLOGY_DIMS = [1]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_PLOT_PATH = _REPO_ROOT / "paper" / "figures" / "replication_gidea2020_btc.png"

# Dates used by the qualitative match (both the single-run replicate and the
# multi-eps sweep). The pre-crisis baseline is the median of L^1 windows
# ending strictly before _PRE_CRISIS_CUTOFF; the two peak windows are the
# ones Gidea et al. highlight in the paper's discussion of Fig. 7c.
_PRE_CRISIS_CUTOFF = pd.Timestamp("2017-09-01", tz="UTC")
_DEC_PEAK_START = pd.Timestamp("2017-12-10", tz="UTC")
_DEC_PEAK_END = pd.Timestamp("2017-12-24", tz="UTC")
_JAN_PEAK_START = pd.Timestamp("2017-12-25", tz="UTC")
_JAN_PEAK_END = pd.Timestamp("2018-01-07", tz="UTC")


# ---------------------------------------------------------------------------
# CryptoCompare fetcher
# ---------------------------------------------------------------------------


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _date_tag(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _cache_path(symbol: str, start: datetime, end: datetime, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / (
        f"{symbol.upper()}USD_cryptocompare_1d_"
        f"{_date_tag(start)}_{_date_tag(end)}.parquet"
    )


def fetch_daily_close(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    use_cache: bool = True,
    cache_root: Path = _DEFAULT_CACHE_ROOT,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch daily USD closes for ``symbol`` over ``[start, end]`` (inclusive).

    Uses CryptoCompare's ``histoday`` endpoint (free, no auth). The result is
    cached as a one-column Parquet under ``cache_root`` keyed by
    ``(symbol, start, end)``.
    """
    start = _to_utc(start)
    end = _to_utc(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")

    path = _cache_path(symbol, start, end, cache_root)
    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        return cached["close"]

    # CryptoCompare paginates backwards from `toTs` returning `limit+1` bars
    # ending at toTs (inclusive). Page size cap is 2000.
    end_ts = int(end.timestamp())
    start_ts = int(start.timestamp())
    seconds_per_day = 86400
    expected_bars = (end_ts - start_ts) // seconds_per_day + 1
    page_size = 2000

    frames: list[pd.DataFrame] = []
    cursor = end_ts
    while True:
        limit = min(page_size, expected_bars + 5)  # tiny over-fetch for safety
        params = {
            "fsym": symbol.upper(),
            "tsym": "USD",
            "limit": limit,
            "toTs": cursor,
        }
        resp = requests.get(CRYPTOCOMPARE_HISTODAY, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("Response") != "Success":
            raise RuntimeError(
                f"CryptoCompare returned non-success for {symbol}: "
                f"{payload.get('Message')!r}"
            )
        rows = payload["Data"]["Data"]
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        oldest = rows[0]["time"]
        if oldest <= start_ts or len(rows) < limit:
            break
        cursor = oldest - seconds_per_day
        # Adjust how many more bars we still need.
        expected_bars = (oldest - start_ts) // seconds_per_day

    if not frames:
        raise RuntimeError(f"no data returned for {symbol}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True).rename("timestamp")
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    # CryptoCompare returns 0.0 close for days the asset did not yet exist.
    df = df[df["close"] > 0]
    out = df[["close"]].astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out["close"]


# ---------------------------------------------------------------------------
# Alignment + log-returns
# ---------------------------------------------------------------------------


def align_log_returns(
    closes: dict[str, pd.Series],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Align daily closes onto a UTC daily grid and return log-returns.

    Mirrors ``data_ingestion.load_aligned_returns``: reindex each series onto
    the inclusive daily grid, intersect via ``dropna(how='any')`` so every
    symbol has coverage, then ``log(close).diff()``.
    """
    start = _to_utc(start)
    end = _to_utc(end)
    grid = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    aligned: dict[str, pd.Series] = {}
    for symbol, series in closes.items():
        s = series.copy()
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        # CryptoCompare timestamps are at 00:00 UTC; normalising guards
        # against any source that pins to end-of-day.
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="first")].reindex(grid)
        aligned[symbol.upper()] = s
    df = pd.DataFrame(aligned).dropna(how="any")
    if df.empty:
        raise RuntimeError("no overlapping dates across symbols after alignment")
    log_returns = np.log(df).diff().dropna(how="any")
    log_returns.index.name = "timestamp"
    return log_returns


# ---------------------------------------------------------------------------
# Replication driver
# ---------------------------------------------------------------------------


@dataclass
class Gidea2020Result:
    """Outputs of the Gidea 2020 Fig. 7 replication."""

    log_returns: pd.DataFrame  # (T, 4)
    btc_close: pd.Series       # BTC close aligned to log-returns dates
    window_end_dates: pd.DatetimeIndex
    l1_h1: np.ndarray
    plot_path: Path
    baseline_median: float
    dec_2017_peak: float
    jan_2018_peak: float

    def qualitative_match(self, peak_ratio_threshold: float = 2.0) -> bool:
        """True iff both peaks dominate the pre-crisis baseline by ``threshold``."""
        if self.baseline_median <= 0 or not np.isfinite(self.baseline_median):
            return False
        dec_ratio = self.dec_2017_peak / self.baseline_median
        jan_ratio = self.jan_2018_peak / self.baseline_median
        return dec_ratio >= peak_ratio_threshold and jan_ratio >= peak_ratio_threshold


def replicate_gidea2020_fig7(
    *,
    symbols: Iterable[str] = GIDEA_SYMBOLS,
    start: datetime = GIDEA_START,
    end: datetime = GIDEA_END,
    window_size: int = GIDEA_WINDOW_SIZE,
    max_edge_length: float = GIDEA_MAX_EDGE_LENGTH,
    homology_dims: Optional[list[int]] = None,
    plot_path: Path = _DEFAULT_PLOT_PATH,
    cache_root: Path = _DEFAULT_CACHE_ROOT,
) -> Gidea2020Result:
    """Reproduce the BTC panel (Fig. 7c) of Gidea et al. 2020.

    Pipeline (paper Section 4):

    1. Daily closes for BTC, ETH, LTC, XRP from 2016-01-01 to 2018-01-07.
    2. Align onto the daily UTC grid; take ``log(close).diff()`` so the
       input is ``(T, 4)`` log-returns.
    3. Sliding window of 50 days, treat each window as a point cloud in
       :math:`\\mathbb{R}^4`, Vietoris-Rips up to
       ``max_edge_length=GIDEA_MAX_EDGE_LENGTH`` (=0.10; see the constant's
       comment for why this differs from the paper's nominal 0.05),
       compute persistence in ``H_1``.
    4. Persistence landscape per window, take :math:`L^1` norm.

    The plot overlays BTC close (top) and :math:`\\|\\lambda\\|_1` per
    window-end date (bottom) and is saved to ``plot_path``.
    """
    if homology_dims is None:
        homology_dims = list(GIDEA_HOMOLOGY_DIMS)
    start = _to_utc(start)
    end = _to_utc(end)
    symbols = tuple(s.upper() for s in symbols)

    logger.info("fetching daily closes for %s from %s to %s", symbols, start.date(), end.date())
    closes = {s: fetch_daily_close(s, start, end, cache_root=cache_root) for s in symbols}

    log_returns = align_log_returns(closes, start, end)
    logger.info("aligned log-returns: shape=%s, range=%s..%s",
                log_returns.shape, log_returns.index.min().date(),
                log_returns.index.max().date())

    pipeline = TDAPipeline(
        mode="multivariate",
        window_size=window_size,
        max_edge_length=max_edge_length,
        homology_dims=homology_dims,
        n_jobs=-1,
    )
    result = pipeline.fit_transform(log_returns.to_numpy())
    l1_h1 = result["L1_H1"]
    n_windows = l1_h1.shape[0]

    # Each window i covers log-return rows i..i+W-1; date-stamp it by the
    # last row's date, matching the paper's "L^1 at end of window" plotting.
    window_end_dates = log_returns.index[window_size - 1 : window_size - 1 + n_windows]
    assert window_end_dates.shape[0] == n_windows

    btc_close = closes["BTC"].copy()
    btc_close.index = btc_close.index.tz_convert("UTC").normalize() \
        if btc_close.index.tz is not None else btc_close.index.tz_localize("UTC").normalize()
    btc_close = btc_close.reindex(log_returns.index).ffill()

    baseline_median, dec_peak, jan_peak = _summarise_peaks(window_end_dates, l1_h1)

    plot_path = Path(plot_path)
    _plot_replication(
        btc_close=btc_close,
        window_end_dates=window_end_dates,
        l1_h1=l1_h1,
        plot_path=plot_path,
    )

    return Gidea2020Result(
        log_returns=log_returns,
        btc_close=btc_close,
        window_end_dates=window_end_dates,
        l1_h1=l1_h1,
        plot_path=plot_path,
        baseline_median=baseline_median,
        dec_2017_peak=dec_peak,
        jan_2018_peak=jan_peak,
    )


# ---------------------------------------------------------------------------
# Plot + summary helpers
# ---------------------------------------------------------------------------


def _summarise_peaks(
    window_end_dates: pd.DatetimeIndex, l1_h1: np.ndarray
) -> tuple[float, float, float]:
    """Pre-crisis baseline + max L^1 in each of the two expected peak windows."""
    series = pd.Series(l1_h1, index=window_end_dates)
    pre_crisis = series.loc[series.index < _PRE_CRISIS_CUTOFF]
    dec_window = series.loc[
        (series.index >= _DEC_PEAK_START) & (series.index <= _DEC_PEAK_END)
    ]
    jan_window = series.loc[
        (series.index >= _JAN_PEAK_START) & (series.index <= _JAN_PEAK_END)
    ]
    baseline = float(pre_crisis.median()) if not pre_crisis.empty else float("nan")
    dec_peak = float(dec_window.max()) if not dec_window.empty else float("nan")
    jan_peak = float(jan_window.max()) if not jan_window.empty else float("nan")
    return baseline, dec_peak, jan_peak


def _plot_replication(
    *,
    btc_close: pd.Series,
    window_end_dates: pd.DatetimeIndex,
    l1_h1: np.ndarray,
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(btc_close.index, btc_close.values, lw=1.0, color="black")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BTC close (USD, log)")
    axes[0].set_title(
        "Replication of Gidea, Goldsmith, Katz, Roldan & Shmalo (2020), Fig. 7c\n"
        "BTC, ETH, LTC, XRP daily log-returns; mode='multivariate', W=50, "
        r"$r_{\max}=0.05$, $H_1$"
    )

    axes[1].plot(window_end_dates, l1_h1, lw=1.0, color="C3")
    axes[1].set_ylabel(r"$\|\lambda\|_1$  (H$_1$)")
    axes[1].set_xlabel("Window end date (UTC)")
    # Mark the two qualitative peaks the paper highlights.
    for date_str, label in (
        ("2017-12-17", "BTC ATH (~Dec 17)"),
        ("2018-01-07", "Jan 7 peak"),
    ):
        d = pd.Timestamp(date_str, tz="UTC")
        if window_end_dates.min() <= d <= window_end_dates.max():
            axes[1].axvline(d, color="grey", lw=0.7, ls="--", alpha=0.7)
            axes[1].text(d, axes[1].get_ylim()[1] * 0.95, " " + label,
                         rotation=90, va="top", ha="left", fontsize=8, color="grey")

    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# max_edge_length sweep
# ---------------------------------------------------------------------------


DEFAULT_EPS_SWEEP: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

_DEFAULT_SWEEP_PLOT_PATH = (
    _REPO_ROOT / "paper" / "figures" / "gidea2020_epsilon_sweep.png"
)


@dataclass
class EpsSweepResult:
    """Outputs of the ``max_edge_length`` sweep on the Gidea 2020 dataset."""

    eps_values: tuple[float, ...]
    window_end_dates: pd.DatetimeIndex
    l1_by_eps: dict[float, np.ndarray]
    summary: pd.DataFrame  # one row per eps, columns: baseline, dec_peak, jan_peak, ratios
    plot_path: Path

    def first_match(self, threshold: float = 2.0) -> Optional[float]:
        """Smallest ``eps`` whose Dec *and* Jan peaks both exceed ``threshold``×baseline."""
        for eps in self.eps_values:
            row = self.summary.loc[self.summary["eps"] == eps].iloc[0]
            if row["dec_ratio"] >= threshold and row["jan_ratio"] >= threshold:
                return float(eps)
        return None


def sweep_max_edge_length(
    *,
    eps_values: Iterable[float] = DEFAULT_EPS_SWEEP,
    symbols: Iterable[str] = GIDEA_SYMBOLS,
    start: datetime = GIDEA_START,
    end: datetime = GIDEA_END,
    window_size: int = GIDEA_WINDOW_SIZE,
    plot_path: Path = _DEFAULT_SWEEP_PLOT_PATH,
    cache_root: Path = _DEFAULT_CACHE_ROOT,
) -> EpsSweepResult:
    """Sweep ``max_edge_length`` on the Gidea 2020 dataset and plot all L^1 series.

    Fetches data once, then re-runs :class:`TDAPipeline` for each ``eps`` in
    ``eps_values``. The per-window L^1 series for H_1 is plotted in its own
    subplot; the two expected peak windows are shaded for visual reference.
    """
    eps_values = tuple(float(e) for e in eps_values)
    start = _to_utc(start)
    end = _to_utc(end)
    symbols = tuple(s.upper() for s in symbols)

    logger.info("sweep: fetching %s daily closes %s..%s",
                symbols, start.date(), end.date())
    closes = {s: fetch_daily_close(s, start, end, cache_root=cache_root) for s in symbols}
    log_returns = align_log_returns(closes, start, end)
    arr = log_returns.to_numpy()
    logger.info("sweep: log-returns shape=%s", arr.shape)

    l1_by_eps: dict[float, np.ndarray] = {}
    window_end_dates: Optional[pd.DatetimeIndex] = None
    summary_rows: list[dict] = []

    for eps in eps_values:
        logger.info("sweep: running TDAPipeline with max_edge_length=%.3f", eps)
        pipeline = TDAPipeline(
            mode="multivariate",
            window_size=window_size,
            max_edge_length=eps,
            homology_dims=[1],
            n_jobs=-1,
        )
        result = pipeline.fit_transform(arr)
        l1 = result["L1_H1"]
        n_windows = l1.shape[0]
        ends = log_returns.index[window_size - 1 : window_size - 1 + n_windows]
        if window_end_dates is None:
            window_end_dates = ends
        l1_by_eps[eps] = l1

        baseline, dec_peak, jan_peak = _summarise_peaks(ends, l1)
        denom = max(baseline, 1e-12)
        summary_rows.append({
            "eps": eps,
            "baseline_median": baseline,
            "dec_peak": dec_peak,
            "jan_peak": jan_peak,
            "dec_ratio": dec_peak / denom,
            "jan_ratio": jan_peak / denom,
        })

    assert window_end_dates is not None  # guaranteed by non-empty eps_values
    summary = pd.DataFrame(summary_rows)

    plot_path = Path(plot_path)
    _plot_eps_sweep(window_end_dates, l1_by_eps, summary, plot_path)

    return EpsSweepResult(
        eps_values=eps_values,
        window_end_dates=window_end_dates,
        l1_by_eps=l1_by_eps,
        summary=summary,
        plot_path=plot_path,
    )


def _plot_eps_sweep(
    window_end_dates: pd.DatetimeIndex,
    l1_by_eps: dict[float, np.ndarray],
    summary: pd.DataFrame,
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    eps_values = list(l1_by_eps.keys())
    n = len(eps_values)
    ncols = 2 if n > 3 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11, 2.4 * nrows), sharex=True, squeeze=False
    )
    flat = axes.flatten()
    for ax, eps in zip(flat, eps_values):
        l1 = l1_by_eps[eps]
        ax.plot(window_end_dates, l1, lw=1.0, color="C3")
        ax.axvspan(_DEC_PEAK_START, _DEC_PEAK_END, alpha=0.12, color="C0",
                   label="Dec 10–24, 2017")
        ax.axvspan(_JAN_PEAK_START, _JAN_PEAK_END, alpha=0.12, color="C2",
                   label="Dec 25, 2017 – Jan 7, 2018")
        row = summary.loc[summary["eps"] == eps].iloc[0]
        ax.set_title(
            rf"$\varepsilon_{{\max}}={eps:.2f}$    "
            rf"baseline={row['baseline_median']:.3g}    "
            rf"Dec/base={row['dec_ratio']:.1f}×    "
            rf"Jan/base={row['jan_ratio']:.1f}×",
            fontsize=9,
        )
        ax.set_ylabel(r"$\|\lambda\|_1$")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # Hide any unused panes.
    for ax in flat[n:]:
        ax.set_visible(False)
    flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Gidea 2020 Fig. 7c replication — "
        r"$\|\lambda\|_1$ (H$_1$) per sliding window across $\varepsilon_{\max}$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = replicate_gidea2020_fig7()
    print()
    print(f"Saved plot: {result.plot_path}")
    print(f"Log-returns shape: {result.log_returns.shape}")
    print(f"Windows: {result.l1_h1.shape[0]}")
    print(f"Pre-crisis L1 median (baseline): {result.baseline_median:.4g}")
    print(f"Dec 10-24, 2017 peak:            {result.dec_2017_peak:.4g}  "
          f"({result.dec_2017_peak / max(result.baseline_median, 1e-12):.2f}x)")
    print(f"Dec 25 2017 - Jan 7, 2018 peak:  {result.jan_2018_peak:.4g}  "
          f"({result.jan_2018_peak / max(result.baseline_median, 1e-12):.2f}x)")
    match = result.qualitative_match()
    print()
    print(f"Qualitative match to Gidea 2020 Fig. 7c: {'YES' if match else 'NO'}")
    return 0 if match else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
