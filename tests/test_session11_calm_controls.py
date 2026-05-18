"""End-to-end tests for ``tda_oct10.analysis_calm_controls`` (Session 11).

The real Binance bulk archive is never hit. For each calm-window
config, a synthetic minute log-returns DataFrame is constructed on
the same UTC grid as the real analysis window so that ``EventConfig``
time arithmetic (calm period, synthetic precursor window, synthetic
center) lines up with the synthetic data. The pipeline is then
exercised end-to-end with ``n_surrogates=10`` per method.

The two tests cover:

1. **Config self-consistency** — both calm configs are well-formed,
   use the Oct 10 basket, have no epicenter symbols, and their
   synthetic-center timestamp is the exact midpoint of the 5-day
   window.
2. **Combined report rendering** — running both calm configs through
   the pipeline (small N) and writing the calm-controls report
   yields a markdown file containing all five rows (three cascade
   references from Session 10 + two new calm rows) and a discussion
   section.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tda_oct10 import analysis_calm_controls as acc  # noqa: E402
from tda_oct10 import analysis_control_events as ace  # noqa: E402
from tda_oct10.analysis_oct10_minute import WINDOW_SIZE  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data on a calm window's grid
# ---------------------------------------------------------------------------


def _synthetic_calm_returns(config: ace.EventConfig, seed: int = 0) -> pd.DataFrame:
    """Iid Gaussian minute returns on ``config``'s grid -- no injected shock.

    A calm window by construction: there is no coordinated cross-asset
    move at the synthetic-center timestamp. The volatility scale
    (``sigma=0.002``) matches real crypto minute volatility so the
    Vietoris-Rips pipeline with ``max_edge_length=0.10`` yields a
    non-trivial mix of zero and non-zero persistence diagrams, but the
    resulting $L^1$ $H_1$ trace should not exhibit a precursor-style
    build-up.
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
    return pd.DataFrame(base, index=grid, columns=list(config.symbols))


def _make_smoke_run(config: ace.EventConfig, tmp_path: Path) -> ace.EventResult:
    returns = _synthetic_calm_returns(config, seed=0)
    main = ace.run_event_main_analysis(returns, config, data_warnings=(), plot=False)
    main.main_plot_path = ace._plot_main_panels(
        config=config,
        log_returns=returns,
        landscape=main.landscape,
        kendall_tau=main.kendall_tau,
        plot_path=tmp_path / f"{config.name}_main.png",
    )
    calm = returns.loc[
        (returns.index >= config.calm_start) & (returns.index < config.calm_end)
    ]
    surrogate = ace.run_event_surrogate_test(
        returns,
        config,
        main,
        n_surrogates=10,
        seed=11,
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
        seed=11,
        plot_path=tmp_path / f"{config.name}_surrogate.png",
    )
    return ace.EventResult(main=main, surrogate=surrogate)


# ---------------------------------------------------------------------------
# Test 1 — calm configs are self-consistent
# ---------------------------------------------------------------------------


def test_calm_event_configs_self_consistent() -> None:
    """Both calm configs are well-formed, use the Oct 10 basket, and the
    synthetic center is at the exact midpoint of the 5-day window."""
    configs = (acc.JULY2024_CALM_CONFIG, acc.MARCH2024_CALM_CONFIG)
    for cfg in configs:
        # Basket is the Oct 10 basket exactly (in the same order).
        assert cfg.symbols == ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE")
        # No epicenter symbols -- this is calm, not a cascade.
        assert cfg.epicenter_symbols == ()
        assert "calm" in cfg.cascade_descriptor.lower()
        assert "calm" in cfg.center_label.lower()
        # The output figures must be the brief-mandated filenames so the
        # paper figure references resolve.
        assert cfg.main_plot_path.name == f"{cfg.name}_main.png"
        assert cfg.surrogate_plot_path.name == f"{cfg.name}_surrogate_test.png"
        # Synthetic center = exact midpoint of the 5-day window.
        midpoint = cfg.analysis_start + (cfg.analysis_end - cfg.analysis_start) / 2
        assert cfg.cascade_center == midpoint.floor("min")
        # Derived windows are the same as for the cascade events: 3h baseline,
        # 360-min precursor strictly before center, calm slice = everything
        # from analysis_start up to precursor_start.
        assert cfg.baseline_end - cfg.baseline_start == pd.Timedelta(hours=3)
        assert cfg.precursor_start < cfg.precursor_end < cfg.cascade_center
        precursor_len = (cfg.precursor_end - cfg.precursor_start).total_seconds() / 60
        assert precursor_len == cfg.pre_cascade_lead_minutes - 1
        assert cfg.calm_start == cfg.analysis_start
        assert cfg.calm_end == cfg.precursor_start
        # Calm slice must have enough rows to support the surrogate test
        # (WINDOW_SIZE + INDICATOR_WINDOW + SURROGATE_BASELINE_HEAD_MINUTES = 290).
        # A 5-day window with a midpoint synthetic center gives ~54 hours of
        # pre-precursor data = ~3240 minutes, comfortably above the floor.
        calm_minutes = (cfg.calm_end - cfg.calm_start).total_seconds() / 60
        assert calm_minutes > 290
    # The two configs target the windows specified in the Session 11 brief.
    assert acc.JULY2024_CALM_CONFIG.analysis_start == pd.Timestamp(
        "2024-07-01 00:00", tz="UTC"
    )
    assert acc.JULY2024_CALM_CONFIG.analysis_end == pd.Timestamp(
        "2024-07-05 23:59", tz="UTC"
    )
    assert acc.MARCH2024_CALM_CONFIG.analysis_start == pd.Timestamp(
        "2024-03-15 00:00", tz="UTC"
    )
    assert acc.MARCH2024_CALM_CONFIG.analysis_end == pd.Timestamp(
        "2024-03-19 23:59", tz="UTC"
    )


# ---------------------------------------------------------------------------
# Test 2 — combined report renders with cascade + calm rows
# ---------------------------------------------------------------------------


def test_calm_controls_report_renders(tmp_path: Path) -> None:
    """Running both calm configs through the pipeline (small N) and writing
    the calm-controls report produces a markdown file with all five rows
    and a discussion section."""
    july = _make_smoke_run(acc.JULY2024_CALM_CONFIG, tmp_path / "july")
    march = _make_smoke_run(acc.MARCH2024_CALM_CONFIG, tmp_path / "march")
    report_path = tmp_path / "calm_controls_findings.md"
    out = acc.write_calm_controls_report(
        results=[july, march], report_path=report_path
    )
    text = out.read_text()

    # Headline mentions Session 11 framing.
    assert "calm" in text.lower()
    assert "precursor" in text.lower()

    # All five comparison rows must be present.
    for cascade_label in ("Oct 10 2025", "Terra-Luna", "FTX"):
        assert cascade_label in text, f"missing cascade row: {cascade_label}"
    for calm_label in ("July 2024", "March 2024"):
        assert calm_label in text, f"missing calm row: {calm_label}"

    # Both nulls referenced in the table.
    assert "Phase-rand p" in text
    assert "Bootstrap p" in text
    # Verdict + discussion sections present.
    assert "## Discussion" in text
    assert "Verdict" in text

    # Synthetic data is iid Gaussian -- the calm-control verdict should be
    # that the precursor signature does NOT fire. Both observed
    # T_run_max values should be small relative to the cascade rows
    # (Terra: 3.750 h, Oct 10: 0.783 h, FTX: 0.333 h).
    for result in (july, march):
        assert result.surrogate.observed_t_run_max < 0.333, (
            f"{result.main.config.name}: synthetic calm window produced an "
            f"unexpectedly large T_run_max = "
            f"{result.surrogate.observed_t_run_max:.3f} h; the test data "
            "may be miscalibrated."
        )

    # Pipeline shape assertions mirroring the Session 10 smoke test.
    for result in (july, march):
        surr = result.surrogate
        assert surr.n_surrogates == 10
        assert surr.phase_t_run_max.shape == (10,)
        assert surr.bootstrap_t_run_max.shape == (10,)
        # Conservative empirical p-value floor with N = 10 is 1 / 11.
        floor = 1.0 / 11.0
        for p in (surr.phase_p_value, surr.bootstrap_p_value):
            assert floor - 1e-12 <= p <= 1.0
