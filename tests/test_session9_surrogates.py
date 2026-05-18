"""End-to-end tests for ``tda_oct10.analysis_oct10_surrogate``.

These tests never hit the network. A synthetic minute-resolution
log-returns DataFrame is constructed on the Oct 10 study grid (so the
calm / precursor / cascade timestamps line up correctly), then piped
through ``run_session9_surrogate_analysis`` with ``n_surrogates=10`` to
exercise the full surrogate-generation → TDA → T_run_max → p-value loop
without the multi-minute compute cost of the production run.
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

from tda_oct10 import analysis_oct10_surrogate as ao9  # noqa: E402
from tda_oct10.analysis_oct10_minute import SYMBOLS_OCT10, WINDOW_SIZE  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data on the real Oct 10 grid
# ---------------------------------------------------------------------------


def _synthetic_oct10_minute_returns(seed: int = 0) -> pd.DataFrame:
    """Return a synthetic minute-returns DataFrame on the real Oct 10 grid.

    The grid runs from one minute past ``CALM_START`` to one minute past
    ``CASCADE_CENTER + 1h`` so that:

    * the *calm* slice [CALM_START, CALM_END) contains enough rows to
      support a 50-min landscape window + 60-min indicator window + 180-min
      baseline head;
    * the *precursor* slice [PRECURSOR_START, PRECURSOR_END] is non-empty
      and aligns with the same grid.

    The returns are iid Gaussian with one coordinated downward shock at
    the cascade-center minute to give the pipeline a non-trivial point
    cloud structure post-cascade. Variance is sized so iid noise plus a
    Vietoris-Rips with max_edge_length=0.10 still produces some non-zero
    persistence diagrams.
    """
    grid = pd.date_range(
        start=ao9.CALM_START + pd.Timedelta(minutes=1),
        end=ao9.CASCADE_CENTER + pd.Timedelta(hours=1),
        freq="min",
        tz="UTC",
        name="timestamp",
    )
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=0.002, size=(len(grid), len(SYMBOLS_OCT10)))
    df = pd.DataFrame(base, index=grid, columns=list(SYMBOLS_OCT10))
    if ao9.CASCADE_CENTER in df.index:
        df.loc[ao9.CASCADE_CENTER] = [-0.04, -0.06, -0.08, -0.06, -0.08, -0.12]
    return df


# ---------------------------------------------------------------------------
# T_run_max statistic
# ---------------------------------------------------------------------------


def test_longest_true_run_basic() -> None:
    """The longest-run helper finds the longest contiguous block of True."""
    assert ao9._longest_true_run(np.array([], dtype=bool)) == 0
    assert ao9._longest_true_run(np.array([False, False, False])) == 0
    assert ao9._longest_true_run(np.array([True])) == 1
    assert ao9._longest_true_run(np.array([True, False, True, True, False, True])) == 2
    # Run at the end.
    assert ao9._longest_true_run(np.array([False, True, True, True])) == 3
    # Run at the start.
    assert ao9._longest_true_run(np.array([True, True, True, False, True])) == 3


def test_t_run_max_handles_invalid_threshold() -> None:
    """A non-positive or non-finite threshold yields NaN, not a spurious run."""
    arr = np.linspace(0.0, 1.0, 200)
    assert np.isnan(ao9._t_run_max(arr, threshold=0.0))
    assert np.isnan(ao9._t_run_max(arr, threshold=float("nan")))
    assert np.isnan(ao9._t_run_max(arr, threshold=-1.0))


def test_t_run_max_units_are_hours() -> None:
    """A constant series above threshold reports the right run length in hours."""
    n_minutes = 200
    arr = np.full(n_minutes, 10.0)
    # 60-min rolling mean of a constant equals the constant; after warm-up
    # (first 59 NaNs) all 200-59 = 141 samples exceed any threshold < 10.
    expected_minutes = n_minutes - (ao9.INDICATOR_WINDOW - 1)
    t = ao9._t_run_max(arr, threshold=1.0)
    assert t == pytest.approx(expected_minutes / 60.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Constants are self-consistent
# ---------------------------------------------------------------------------


def test_precursor_window_strictly_before_cascade() -> None:
    """The observed precursor window must end strictly before the cascade."""
    assert ao9.PRECURSOR_END < pd.Timestamp("2025-10-10 21:15", tz="UTC")
    assert ao9.PRECURSOR_START < ao9.PRECURSOR_END
    assert ao9.CALM_START < ao9.CALM_END
    assert ao9.CALM_END <= ao9.PRECURSOR_START


# ---------------------------------------------------------------------------
# End-to-end pipeline + surrogates (10 surrogates each method)
# ---------------------------------------------------------------------------


def test_run_session9_end_to_end_small(tmp_path: Path) -> None:
    """Full surrogate analysis runs without error on 10 synthetic surrogates each method."""
    returns = _synthetic_oct10_minute_returns(seed=0)
    # Sanity: calm slice has enough rows to support the W=50 + W_ind=60 +
    # head=180 minimum.
    calm = returns.loc[(returns.index >= ao9.CALM_START) & (returns.index < ao9.CALM_END)]
    assert calm.shape[0] > WINDOW_SIZE + ao9.INDICATOR_WINDOW + ao9.SURROGATE_BASELINE_HEAD_MINUTES

    figure_path = tmp_path / "session9_test.png"
    report_path = tmp_path / "session9_test.md"

    result = ao9.run_session9_surrogate_analysis(
        log_returns=returns,
        returns_path=None,
        n_surrogates=10,
        seed=123,
        n_jobs=1,                # keep the test serial — joblib parallel adds startup cost
        figure_path=figure_path,
        report_path=report_path,
        n_display=5,
    )

    # Structural assertions on the bundled result.
    assert result.n_surrogates == 10
    assert np.isfinite(result.observed_baseline)
    assert result.observed_baseline > 0.0
    assert np.isfinite(result.observed_t_run_max)
    assert result.observed_t_run_max >= 0.0

    for run in (result.phase, result.bootstrap):
        assert run.t_run_max.shape == (10,)
        assert run.baselines.shape == (10,)
        assert run.l1_h1_array.shape[0] == 10
        # T_run_max statistic is a finite non-negative duration in hours.
        assert np.all(np.isfinite(run.t_run_max[~np.isnan(run.t_run_max)]))
        assert np.all(run.t_run_max[~np.isnan(run.t_run_max)] >= 0.0)
        # p-value uses the conservative (1+k)/(1+N) convention; floor is 1/11.
        assert run.p_value >= 1.0 / (1.0 + 10.0) - 1e-12
        assert run.p_value <= 1.0

    # Methods differ in random draws and statistic, so they should not produce
    # identical surrogate L^1 H_1 arrays.
    assert not np.array_equal(
        result.phase.l1_h1_array, result.bootstrap.l1_h1_array
    )

    # Outputs were materialised.
    assert figure_path.exists() and figure_path.stat().st_size > 0
    assert report_path.exists() and report_path.stat().st_size > 0
    text = report_path.read_text()
    assert "Phase-randomized" in text
    assert "bootstrap" in text.lower()
    assert "p-value" in text.lower() or "p =" in text


def test_calm_period_too_short_raises() -> None:
    """A pathologically small log_returns frame fails with a clear error."""
    grid = pd.date_range(
        start=ao9.CALM_START + pd.Timedelta(minutes=1),
        periods=50,
        freq="min",
        tz="UTC",
        name="timestamp",
    )
    df = pd.DataFrame(
        np.zeros((50, len(SYMBOLS_OCT10))),
        index=grid,
        columns=list(SYMBOLS_OCT10),
    )
    with pytest.raises(ValueError, match="calm period"):
        ao9.run_session9_surrogate_analysis(
            log_returns=df,
            returns_path=None,
            n_surrogates=3,
            n_jobs=1,
            figure_path=None,
            report_path=None,
        )
