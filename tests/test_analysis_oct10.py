"""End-to-end tests for ``tda_oct10.analysis_oct10``.

The real CryptoCompare/Binance fetch is never exercised here — the test
constructs a synthetic returns DataFrame on the same daily grid as the
Oct 10 study window and pipes it through :func:`run_daily_analysis`.

The goal is to lock in the *shape* contract of the daily pipeline, not
to validate numerical accuracy (that is the job of ``test_tda_pipeline``,
``test_indicators_and_trend``, and the Gidea replication in
``tda_oct10.validation``). Concretely:

* the 50-day landscape window over N daily returns yields N - 49 rows;
* the 20-day indicator window over those yields N - 49 - 19 rows;
* the 10-day Kendall window over those yields N - 49 - 19 - 9 rows.

For N = 121 (122 daily closes between 2025-08-01 and 2025-11-30) those
counts are 72, 53, and 44 respectively.
"""

from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tda_oct10 import analysis_oct10 as ao  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _synthetic_returns(seed: int = 0) -> pd.DataFrame:
    """Return a 121x6 daily log-returns frame on the real Oct-10 study grid.

    The series uses ``numpy.random.default_rng`` so it is deterministic
    and free of any networked dependency. A single large coordinated
    move is injected on 2025-10-10 so the cascade shape — and the
    summary-table machinery — exercises a non-trivial code path.
    """
    grid = pd.date_range(
        start="2025-08-02",
        end="2025-11-30",
        freq="D",
        tz="UTC",
        name="timestamp",
    )
    assert len(grid) == 121, f"expected 121 rows, got {len(grid)}"
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=0.02, size=(len(grid), len(ao.SYMBOLS_OCT10)))
    df = pd.DataFrame(base, index=grid, columns=list(ao.SYMBOLS_OCT10))
    # Inject the cascade shape on Oct 10.
    df.loc[ao.OCT10_DATE] = [-0.08, -0.13, -0.16, -0.13, -0.17, -0.24]
    return df


def _synthetic_closes(returns: pd.DataFrame) -> dict[str, pd.Series]:
    """Reconstruct nominal close series from log-returns for plot panel 1."""
    out: dict[str, pd.Series] = {}
    for col in returns.columns:
        prices = 100.0 * np.exp(returns[col].cumsum())
        out[col] = prices
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_daily_analysis_end_to_end_shapes(tmp_path: Path) -> None:
    """Pipeline runs without error on synthetic data and yields the documented shapes."""
    returns = _synthetic_returns()
    closes = _synthetic_closes(returns)

    plot_path = tmp_path / "fig.png"
    report_path = tmp_path / "report.md"
    returns_path = tmp_path / "returns.parquet"

    result = ao.run_daily_analysis(
        log_returns=returns,
        closes=closes,
        returns_path=returns_path,
        plot_path=plot_path,
        report_path=report_path,
    )

    # Shape contract: 121 returns → 72 landscape rows → 53 indicator rows →
    # 44 Kendall tau rows.
    n = len(returns)
    assert n == 121
    expected_landscape = n - ao.WINDOW_SIZE + 1
    expected_indicators = expected_landscape - ao.INDICATOR_WINDOW + 1
    expected_kendall = expected_indicators - ao.KENDALL_WINDOW + 1
    assert expected_landscape == 72
    assert expected_indicators == 53
    assert expected_kendall == 44

    assert result.log_returns.shape == returns.shape
    assert result.landscape.shape == (expected_landscape, 5)
    assert list(result.landscape.columns) == ["L1_H0", "L1_H1", "L2_H0", "L2_H1", "PE_H1"]
    assert result.indicators.shape == (expected_indicators, 3)
    assert list(result.indicators.columns) == ["AC1", "VAR", "MPS"]
    assert result.kendall_tau.shape == (expected_kendall, 6)
    assert set(result.kendall_tau.columns) == {
        "tau_AC1", "tau_VAR", "tau_MPS", "p_AC1", "p_VAR", "p_MPS",
    }

    # Window-end dates are monotonic and aligned with the input grid.
    assert result.window_end_dates.is_monotonic_increasing
    assert result.window_end_dates[0] == returns.index[ao.WINDOW_SIZE - 1]
    assert result.window_end_dates[-1] == returns.index[-1]

    # Outputs were materialised.
    assert returns_path.exists()
    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert report_path.exists() and report_path.stat().st_size > 0
    assert result.plot_path == plot_path
    assert result.report_path == report_path

    # The Oct 10 cascade row shows up in the summary table with finite
    # L^1 / L^2 values, and the post-cascade L^1 H_1 exceeds the
    # pre-cascade baseline median (a sanity check that the H_1 signal
    # tracks the injected move).
    summary = result.summary_table
    assert {"L1_H0", "L1_H1", "L2_H0", "L2_H1", "PE_H1"}.issubset(summary.index)
    cascade_l1_h1 = float(summary.loc["L1_H1", "on_2025_10_10"])
    assert np.isfinite(cascade_l1_h1)
    post_max_l1_h1 = float(
        result.landscape.loc[result.landscape.index >= ao.OCT10_DATE, "L1_H1"].max()
    )
    baseline_l1_h1 = float(summary.loc["L1_H1", "baseline_median"])
    if np.isfinite(baseline_l1_h1) and baseline_l1_h1 > 0:
        assert post_max_l1_h1 > baseline_l1_h1


def test_run_daily_analysis_skips_outputs_when_paths_none() -> None:
    """``None`` paths leave no side effects on disk and skip the plot."""
    returns = _synthetic_returns(seed=7)
    result = ao.run_daily_analysis(
        log_returns=returns,
        closes={},                  # no BTC/ETH → plot must be skipped
        returns_path=None,
        plot_path=None,
        report_path=None,
    )
    assert result.plot_path is None
    assert result.report_path is None
    assert result.landscape.shape[0] == len(returns) - ao.WINDOW_SIZE + 1


def test_format_summary_table_has_four_columns() -> None:
    """The pretty-printer renders all four reference columns."""
    returns = _synthetic_returns(seed=3)
    result = ao.run_daily_analysis(
        log_returns=returns,
        closes={},
        returns_path=None,
        plot_path=None,
        report_path=None,
    )
    text = ao.format_summary_table(result.summary_table)
    for header in (
        "baseline (Aug 15–Oct 5)",
        "2025-10-03 (week before)",
        "2025-10-10 (cascade)",
        "2025-10-17 (week after)",
    ):
        assert header in text
    for metric in ("L1_H0", "L1_H1", "L2_H0", "L2_H1", "PE_H1", "tau_VAR"):
        assert metric in text
