"""End-to-end tests for ``tda_oct10.analysis_oct10_minute``.

The real Binance bulk archive is never hit. The test constructs a
synthetic minute-resolution log-returns DataFrame on the same UTC grid
as the Oct 10 study window and pipes it through :func:`run_minute_analysis`.

Shape contract (mirrors the daily test but at minute scale):

* a 50-minute landscape window over N rows yields ``N - 49`` rows;
* a 60-minute indicator window over those yields ``N - 49 - 59`` rows;
* a 30-minute Kendall window over those yields ``N - 49 - 59 - 29`` rows.
"""

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

from tda_oct10 import analysis_oct10_minute as am  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _synthetic_minute_returns(n_minutes: int = 720, seed: int = 0) -> pd.DataFrame:
    """Return a ``(n_minutes, 6)`` minute log-returns frame on the Oct-10 grid.

    A coordinated downward shock is injected at the cascade-center minute
    so the pipeline exercises the same non-trivial code path as the real
    cascade does. ``n_minutes=720`` (12 hours) keeps the test fast while
    still producing well-defined indicator and Kendall-tau series.
    """
    grid = pd.date_range(
        start="2025-10-10 15:15",
        periods=n_minutes,
        freq="min",
        tz="UTC",
        name="timestamp",
    )
    assert grid[0] == pd.Timestamp("2025-10-10 15:15", tz="UTC")
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=0.001, size=(n_minutes, len(am.SYMBOLS_OCT10)))
    df = pd.DataFrame(base, index=grid, columns=list(am.SYMBOLS_OCT10))
    if am.CASCADE_CENTER in df.index:
        df.loc[am.CASCADE_CENTER] = [-0.04, -0.06, -0.08, -0.06, -0.08, -0.12]
    return df


def _synthetic_closes(returns: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for col in ("BTC", "ETH"):
        prices = 100.0 * np.exp(returns[col].cumsum())
        out[col] = prices
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_minute_analysis_end_to_end_shapes(tmp_path: Path) -> None:
    """Pipeline runs without error on synthetic minute data and yields documented shapes."""
    returns = _synthetic_minute_returns(n_minutes=720, seed=0)
    closes = _synthetic_closes(returns)

    main_plot = tmp_path / "main.png"
    context_plot = tmp_path / "context.png"
    report = tmp_path / "report.md"
    returns_path = tmp_path / "returns.parquet"

    result = am.run_minute_analysis(
        log_returns=returns,
        closes=closes,
        returns_path=returns_path,
        main_plot_path=main_plot,
        context_plot_path=context_plot,
        report_path=report,
    )

    # Shape contract: 720 returns → 671 landscape rows → 612 indicator rows
    # → 583 Kendall tau rows.
    n = len(returns)
    expected_landscape = n - am.WINDOW_SIZE + 1
    expected_indicators = expected_landscape - am.INDICATOR_WINDOW + 1
    expected_kendall = expected_indicators - am.KENDALL_WINDOW + 1
    assert expected_landscape == 671
    assert expected_indicators == 612
    assert expected_kendall == 583

    assert result.log_returns.shape == returns.shape
    assert result.landscape.shape == (expected_landscape, 5)
    assert list(result.landscape.columns) == [
        "L1_H0", "L1_H1", "L2_H0", "L2_H1", "PE_H1",
    ]
    assert result.indicators.shape == (expected_indicators, 3)
    assert list(result.indicators.columns) == ["AC1", "VAR", "MPS"]
    assert result.kendall_tau.shape == (expected_kendall, 6)
    assert set(result.kendall_tau.columns) == {
        "tau_AC1", "tau_VAR", "tau_MPS", "p_AC1", "p_VAR", "p_MPS",
    }

    # Window-end times are monotonic and aligned with the input grid.
    assert result.window_end_times.is_monotonic_increasing
    assert result.window_end_times[0] == returns.index[am.WINDOW_SIZE - 1]
    assert result.window_end_times[-1] == returns.index[-1]

    # Outputs were materialised.
    assert returns_path.exists()
    assert main_plot.exists() and main_plot.stat().st_size > 0
    assert context_plot.exists() and context_plot.stat().st_size > 0
    assert report.exists() and report.stat().st_size > 0
    assert result.main_plot_path == main_plot
    assert result.context_plot_path == context_plot
    assert result.report_path == report

    # Sanity: the cascade-time L^1 H_1 should exceed the median of the
    # series, given the coordinated shock injected at 21:15 UTC.
    cascade_window_mask = (
        (result.landscape.index >= am.CASCADE_CENTER)
        & (result.landscape.index <= am.CASCADE_CENTER + pd.Timedelta(minutes=30))
    )
    if cascade_window_mask.any():
        cascade_max = float(result.landscape.loc[cascade_window_mask, "L1_H1"].max())
        median = float(result.landscape["L1_H1"].median())
        if np.isfinite(median) and median > 0:
            assert cascade_max >= median


def test_run_minute_analysis_skips_outputs_when_paths_none() -> None:
    """``None`` paths produce no side effects and skip the plots."""
    returns = _synthetic_minute_returns(n_minutes=300, seed=7)
    result = am.run_minute_analysis(
        log_returns=returns,
        closes={},                # no BTC/ETH → plots are skipped
        returns_path=None,
        main_plot_path=None,
        context_plot_path=None,
        report_path=None,
    )
    assert result.main_plot_path is None
    assert result.context_plot_path is None
    assert result.report_path is None
    assert result.landscape.shape[0] == len(returns) - am.WINDOW_SIZE + 1


def test_pre_cascade_window_excludes_cascade_center() -> None:
    """The strict pre-cascade window must end *before* 21:15 UTC."""
    # The window may legitimately end exactly one minute before the
    # cascade, but never at or after it.
    assert am.PRE_CASCADE_END < am.CASCADE_CENTER
    assert am.PRE_CASCADE_START < am.PRE_CASCADE_END
    span_minutes = (am.PRE_CASCADE_END - am.PRE_CASCADE_START).total_seconds() / 60.0
    assert span_minutes <= am.PRE_CASCADE_LEAD_MINUTES
