"""End-to-end tests for ``tda_oct10.analysis_control_events`` (Session 10).

The real Binance bulk archive is never hit. For each control event,
a synthetic minute log-returns DataFrame is constructed on the same
UTC grid as the real analysis window so that ``EventConfig`` time
arithmetic (calm period, precursor window, cascade center) lines up
with the synthetic data. The pipeline is then exercised end-to-end
with ``n_surrogates=10`` per method, asserting the full main +
surrogate workflow runs without error and produces well-shaped
outputs.
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

from tda_oct10 import analysis_control_events as ace  # noqa: E402
from tda_oct10.analysis_oct10_minute import WINDOW_SIZE  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data on a real event's grid
# ---------------------------------------------------------------------------


def _synthetic_event_returns(config: ace.EventConfig, seed: int = 0) -> pd.DataFrame:
    """Iid Gaussian minute returns on ``config``'s grid, with a shock at the cascade.

    Returns are scaled (``sigma=0.002``) to match real crypto minute
    volatility so the Vietoris-Rips pipeline with ``max_edge_length=0.10``
    yields a non-trivial mix of zero and non-zero persistence diagrams.
    A coordinated downward shock is injected at the cascade-center minute
    in every symbol so the "cascade window" assertions in the main
    analysis have something to detect.
    """
    grid = pd.date_range(
        start=config.analysis_start + pd.Timedelta(minutes=1),
        end=config.analysis_end,
        freq="min",
        tz="UTC",
        name="timestamp",
    )
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=0.002, size=(len(grid), len(config.symbols)))
    df = pd.DataFrame(base, index=grid, columns=list(config.symbols))
    if config.cascade_center in df.index:
        # Symmetric coordinated downward shock across the basket.
        df.loc[config.cascade_center] = [-0.04, -0.06, -0.08, -0.06, -0.08, -0.12]
    return df


def _make_smoke_run(config: ace.EventConfig, tmp_path: Path) -> ace.EventResult:
    returns = _synthetic_event_returns(config, seed=0)
    # Sanity: calm slice big enough to support W=50 + W_ind=60 + head=180.
    calm = returns.loc[
        (returns.index >= config.calm_start) & (returns.index < config.calm_end)
    ]
    assert calm.shape[0] > (WINDOW_SIZE + 60 + 180)

    # Redirect outputs to tmp_path by monkey-patching the paths on the config
    # would mutate the frozen dataclass; instead, write our own paths via
    # the lower-level entry points so the test stays hermetic.
    main = ace.run_event_main_analysis(returns, config, data_warnings=(), plot=False)
    # Re-plot to the temp location so we exercise the plot path without
    # writing into paper/figures/.
    main.main_plot_path = ace._plot_main_panels(
        config=config,
        log_returns=returns,
        landscape=main.landscape,
        kendall_tau=main.kendall_tau,
        plot_path=tmp_path / f"{config.name}_main.png",
    )
    surrogate = ace.run_event_surrogate_test(
        returns,
        config,
        main,
        n_surrogates=10,
        seed=7,
        n_jobs=1,
        plot=False,
        n_display=5,
    )
    surrogate.surrogate_plot_path = ace._plot_surrogate_panels(
        config=config,
        surrogate=surrogate,
        calm_index=calm.index,
        real_l1_h1=main.landscape["L1_H1"],
        n_display=5,
        seed=7,
        plot_path=tmp_path / f"{config.name}_surrogate.png",
    )
    return ace.EventResult(main=main, surrogate=surrogate)


# ---------------------------------------------------------------------------
# Config self-consistency
# ---------------------------------------------------------------------------


def test_event_config_windows_are_self_consistent() -> None:
    """Derived windows (baseline, precursor, calm) line up correctly."""
    for cfg in (ace.TERRA_LUNA_CONFIG, ace.FTX_CONFIG):
        # Baseline is 3h at the start.
        assert cfg.baseline_end - cfg.baseline_start == pd.Timedelta(hours=3)
        # Precursor is `pre_cascade_lead_minutes` long, strictly before cascade.
        assert cfg.precursor_start < cfg.precursor_end < cfg.cascade_center
        precursor_len = (cfg.precursor_end - cfg.precursor_start).total_seconds() / 60
        assert precursor_len == cfg.pre_cascade_lead_minutes - 1
        # Calm period is the analysis-start head, ending strictly before the precursor.
        assert cfg.calm_start == cfg.analysis_start
        assert cfg.calm_end == cfg.precursor_start


def test_event_config_basket_size_is_six() -> None:
    """The Session 8 pipeline assumes a 6-symbol basket; controls match."""
    assert len(ace.TERRA_LUNA_CONFIG.symbols) == 6
    assert len(ace.FTX_CONFIG.symbols) == 6
    assert "LUNA" in ace.TERRA_LUNA_CONFIG.symbols
    assert "UST" in ace.TERRA_LUNA_CONFIG.symbols
    assert "FTT" in ace.FTX_CONFIG.symbols


# ---------------------------------------------------------------------------
# Per-event end-to-end smoke tests
# ---------------------------------------------------------------------------


def test_terra_luna_end_to_end_small(tmp_path: Path) -> None:
    """Synthetic Terra-Luna minute returns run through main + surrogate without error."""
    result = _make_smoke_run(ace.TERRA_LUNA_CONFIG, tmp_path)
    assert result.main.config.name == "terra_luna"
    assert result.main.landscape.shape[0] == result.main.log_returns.shape[0] - WINDOW_SIZE + 1
    assert set(result.main.landscape.columns) >= {"L1_H0", "L1_H1", "L2_H0", "L2_H1"}
    assert np.isfinite(result.main.baseline_l1_h1) and result.main.baseline_l1_h1 > 0
    assert np.isfinite(result.main.baseline_l1_h0) and result.main.baseline_l1_h0 > 0

    surr = result.surrogate
    assert surr.n_surrogates == 10
    assert surr.phase_t_run_max.shape == (10,)
    assert surr.bootstrap_t_run_max.shape == (10,)
    assert surr.phase_l1_h1_array.shape[0] == 10
    assert surr.bootstrap_l1_h1_array.shape[0] == 10
    # Floor of conservative empirical p-value with N=10 is 1/11.
    floor = 1.0 / 11.0
    for p in (surr.phase_p_value, surr.bootstrap_p_value):
        assert floor - 1e-12 <= p <= 1.0

    assert (tmp_path / "terra_luna_main.png").stat().st_size > 0
    assert (tmp_path / "terra_luna_surrogate.png").stat().st_size > 0


def test_ftx_end_to_end_small(tmp_path: Path) -> None:
    """Synthetic FTX minute returns run through main + surrogate without error."""
    result = _make_smoke_run(ace.FTX_CONFIG, tmp_path)
    assert result.main.config.name == "ftx"
    assert result.main.landscape.shape[0] == result.main.log_returns.shape[0] - WINDOW_SIZE + 1
    # Pipeline reproduces basket order.
    assert tuple(result.main.log_returns.columns) == ace.FTX_CONFIG.symbols

    surr = result.surrogate
    assert surr.phase_l1_h1_array.shape == surr.bootstrap_l1_h1_array.shape
    # The two surrogate families should produce *different* L^1 H_1 arrays
    # (different random draws + different statistical structure).
    assert not np.array_equal(surr.phase_l1_h1_array, surr.bootstrap_l1_h1_array)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_combined_report_renders(tmp_path: Path) -> None:
    """The combined comparison report mentions all three events and both nulls."""
    terra = _make_smoke_run(ace.TERRA_LUNA_CONFIG, tmp_path / "terra")
    ftx = _make_smoke_run(ace.FTX_CONFIG, tmp_path / "ftx")
    report_path = tmp_path / "control_events_findings.md"
    out = ace.write_control_events_report(results=[terra, ftx], report_path=report_path)
    text = out.read_text()
    assert "Oct 10 2025" in text
    assert "Terra-Luna" in text
    assert "FTX" in text
    assert "phase-rand" in text.lower() or "phase-randomized" in text.lower()
    assert "bootstrap" in text.lower()
    assert "p =" in text or "p-value" in text.lower()
    # Discussion section must be present.
    assert "## Discussion" in text
