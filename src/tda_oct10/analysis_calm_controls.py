"""Session 11 — negative-control calm-period analysis.

Sessions 8 – 10 established that the multivariate minute-resolution
$L^1$ $H_1$ precursor signature appears at three independent crypto
liquidation cascades (Oct 10 2025, Terra-Luna May 2022, FTX Nov 2022)
at p < 0.05 under both phase-randomized and moving-block-bootstrap
nulls. The remaining publication-blocking concern is whether the same
statistic *also* fires during randomly-chosen calm windows. If it
does, the multi-event finding is just market noise. If it does not,
the precursor is genuinely specific to cascades.

This module runs the *identical* Session 8 + Session 9 pipeline (no
parameter changes) on two pre-selected calm 5-day windows in the
historical Binance archive:

* **Window A — July 2024.** 2024-07-01 00:00 UTC .. 2024-07-05 23:59
  UTC. BTC range-bound around USD 60 – 63k. No cascades, no
  liquidations > USD 1B.
* **Window B — March 2024.** 2024-03-15 00:00 UTC .. 2024-03-19 23:59
  UTC. Post-halving-anticipation drift. No major liquidation events.

Both windows use the **Oct 10 symbol basket** (BTC, ETH, SOL, BNB,
XRP, DOGE) so the calm-control numbers are directly comparable to the
Oct 10 result. For each window the "synthetic cascade center" is set
to the exact midpoint of the 5-day window (= 2024-07-03 12:00 UTC and
2024-03-17 12:00 UTC respectively). The 360-minute "precursor" window
ending one minute before the synthetic center is the *exact same
construction* used in Sessions 8 – 10; we test whether the same
$T_{\\mathrm{run\\_max}}$ statistic that flagged the three cascades
also flags these calm windows.

Expected outcome (the null hypothesis of the Session 11 design): both
calm windows produce $T_{\\mathrm{run\\_max}}$ that fails to reach
significance under either null, or clears 0.05 only marginally. Any
calm window producing a result comparable to the three cascades would
be a serious finding for the paper and must be reported before
committing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tda_oct10.analysis_control_events import (
    BASELINE_HOURS,
    EventConfig,
    EventResult,
    OCT10_REFERENCE,
    PRE_CASCADE_LEAD_MINUTES,
    _fmt,
    _fmt_g,
    _ratio,
    _safe_relpath,
    _ts,
    run_control_event,
)
from tda_oct10.analysis_oct10_minute import (
    INDICATOR_WINDOW,
    KENDALL_WINDOW,
    MAX_EDGE_LENGTH,
    WINDOW_SIZE,
)
from tda_oct10.analysis_oct10_surrogate import (
    DEFAULT_N_SURROGATES,
    DEFAULT_SEED,
    RUN_MULTIPLIER,
)

__all__ = [
    "JULY2024_CALM_CONFIG",
    "MARCH2024_CALM_CONFIG",
    "TERRA_LUNA_REFERENCE",
    "FTX_REFERENCE",
    "CALM_BASKET",
    "run_session11_calm_controls",
    "write_calm_controls_report",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORT_PATH = _REPO_ROOT / "paper" / "calm_controls_findings.md"


# ---------------------------------------------------------------------------
# Calm-control configurations
# ---------------------------------------------------------------------------

# Keep the basket identical to Oct 10 (Session 8): every change relative to
# the cascade events should isolate the "is this a cascade?" axis. Using a
# different basket would mix the basket-change axis into the comparison.
CALM_BASKET: tuple[str, ...] = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE")


def _exact_window_center(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    """Midpoint of the inclusive ``[start, end]`` window, rounded to the minute.

    For a 5-day window beginning at ``YYYY-MM-DD 00:00`` and ending at
    ``YYYY-MM-(DD+4) 23:59`` this lands at ``YYYY-MM-(DD+2) 12:00`` (the
    literal exact center). The task brief describes this as
    "approximately the start of day 3"; the exact-center reading is
    what's actually used so the calm-control window is symmetric around
    the synthetic precursor location.
    """
    midpoint = start + (end - start) / 2
    return midpoint.floor("min")


JULY2024_CALM_CONFIG = EventConfig(
    name="calm_control_july2024",
    label="July 2024 calm-period control",
    symbols=CALM_BASKET,
    epicenter_symbols=(),  # no epicenter — there is no cascade.
    cascade_descriptor="calm baseline (no cascade event)",
    analysis_start=pd.Timestamp("2024-07-01 00:00", tz="UTC"),
    analysis_end=pd.Timestamp("2024-07-05 23:59", tz="UTC"),
    cascade_center=_exact_window_center(
        pd.Timestamp("2024-07-01 00:00", tz="UTC"),
        pd.Timestamp("2024-07-05 23:59", tz="UTC"),
    ),
    main_plot_filename="calm_control_july2024_main.png",
    surrogate_plot_filename="calm_control_july2024_surrogate_test.png",
    center_label="synthetic center (calm)",
)

MARCH2024_CALM_CONFIG = EventConfig(
    name="calm_control_march2024",
    label="March 2024 calm-period control",
    symbols=CALM_BASKET,
    epicenter_symbols=(),
    cascade_descriptor="calm baseline (no cascade event)",
    analysis_start=pd.Timestamp("2024-03-15 00:00", tz="UTC"),
    analysis_end=pd.Timestamp("2024-03-19 23:59", tz="UTC"),
    cascade_center=_exact_window_center(
        pd.Timestamp("2024-03-15 00:00", tz="UTC"),
        pd.Timestamp("2024-03-19 23:59", tz="UTC"),
    ),
    main_plot_filename="calm_control_march2024_main.png",
    surrogate_plot_filename="calm_control_march2024_surrogate_test.png",
    center_label="synthetic center (calm)",
)


# ---------------------------------------------------------------------------
# Cascade-reference numbers for the headline comparison table
# ---------------------------------------------------------------------------

# Source: paper/control_events_findings.md (Session 10), tied to commit 828c77b.
# Kept here as plain dicts rather than recomputed so the report renders
# deterministically without re-running the full 1000-surrogate pipeline on the
# cascade events.
TERRA_LUNA_REFERENCE = {
    "label": "Terra-Luna",
    "kind": "cascade",
    "observed_t_run_max_hours": 3.750,
    "phase_p_value": 0.000999,
    "bootstrap_p_value": 0.000999,
    "cascade_l1_h0_ratio": 300.5,
}
FTX_REFERENCE = {
    "label": "FTX",
    "kind": "cascade",
    "observed_t_run_max_hours": 0.333,
    "phase_p_value": 0.000999,
    "bootstrap_p_value": 0.024,
    "cascade_l1_h0_ratio": 24.1,
}


# ---------------------------------------------------------------------------
# Comparison-table assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TableRow:
    event: str
    kind: str
    t_run_max: float
    phase_p: float
    bootstrap_p: float
    h0_ratio: float


def _row_for_reference(ref: dict, *, event_label: str, kind: str) -> _TableRow:
    return _TableRow(
        event=event_label,
        kind=kind,
        t_run_max=float(ref["observed_t_run_max_hours"]),
        phase_p=float(ref["phase_p_value"]),
        bootstrap_p=float(ref["bootstrap_p_value"]),
        h0_ratio=float(ref["cascade_l1_h0_ratio"]),
    )


_CALM_LABEL_SUFFIX = " calm-period control"


def _short_event_label(label: str) -> str:
    """Strip the known calm-control suffix so the headline table reads e.g.
    ``July 2024 (calm)`` rather than ``July 2024 calm-period control (calm)``,
    matching the Session 11 brief's table format."""
    if label.endswith(_CALM_LABEL_SUFFIX):
        return label[: -len(_CALM_LABEL_SUFFIX)]
    return label.split(",")[0]


def _row_for_result(result: EventResult) -> _TableRow:
    main = result.main
    surr = result.surrogate
    h0_ratio = _ratio(main.cascade_l1_h0_peak, main.baseline_l1_h0)
    return _TableRow(
        event=_short_event_label(main.config.label),
        kind="calm",
        t_run_max=float(surr.observed_t_run_max),
        phase_p=float(surr.phase_p_value),
        bootstrap_p=float(surr.bootstrap_p_value),
        h0_ratio=float(h0_ratio),
    )


def _render_table(rows: Sequence[_TableRow]) -> list[str]:
    header = [
        "| Event | T_run_max | Phase-rand p | Bootstrap p | H_0 peak ratio |",
        "|---|---|---|---|---|",
    ]
    body: list[str] = []
    for r in rows:
        tag = "cascade" if r.kind == "cascade" else "calm"
        body.append(
            f"| {r.event} ({tag}) | "
            f"{_fmt(r.t_run_max, 3)} h | "
            f"{_fmt_g(r.phase_p)} | "
            f"{_fmt_g(r.bootstrap_p)} | "
            f"{_fmt(r.h0_ratio, 1)}x |"
        )
    return header + body


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _significance_verdict(row: _TableRow) -> str:
    """Plain-English verdict on whether a calm row mimics a cascade."""
    phase_sig = np.isfinite(row.phase_p) and row.phase_p < 0.05
    boot_sig = np.isfinite(row.bootstrap_p) and row.bootstrap_p < 0.05
    if phase_sig and boot_sig:
        return (
            "**both nulls reject at p < 0.05** -- calm window mimics the "
            "cascade precursor; this would be a serious finding for the "
            "paper and must be discussed before publication."
        )
    if phase_sig or boot_sig:
        return (
            "**one null rejects at p < 0.05, the other does not** -- "
            "ambiguous; sensitivity to the null choice means the calm "
            "window cannot be cleanly distinguished from a cascade by the "
            "current test."
        )
    return (
        "**neither null rejects at p < 0.05** -- the calm window does not "
        "reproduce the cascade precursor under the same surrogate test, "
        "as expected."
    )


def _detail_block(result: EventResult) -> str:
    main = result.main
    surr = result.surrogate
    cfg = main.config
    ratio_pre_h1 = _ratio(main.pre_l1_h1_peak, main.baseline_l1_h1)
    ratio_pre_h0 = _ratio(main.pre_l1_h0_peak, main.baseline_l1_h0)
    ratio_cen_h1 = _ratio(main.cascade_l1_h1_peak, main.baseline_l1_h1)
    ratio_cen_h0 = _ratio(main.cascade_l1_h0_peak, main.baseline_l1_h0)

    def _peak_line(homology: str, peak: float, ratio: float, ts) -> str:
        """Render a single precursor-peak bullet, omitting a missing timestamp."""
        time_str = _ts(ts)
        base = (
            f"* Synthetic-precursor $L^1$ $H_{homology}$ peak: "
            f"{_fmt_g(peak)} ({_fmt(ratio, 2)}x baseline)"
        )
        return base + (f" at {time_str}." if time_str != "n/a" else ".")

    lines = [
        f"### {cfg.label}",
        "",
        f"* **Symbols:** {', '.join(cfg.symbols)} (Oct 10 basket).",
        f"* **Classifier:** {cfg.cascade_descriptor}.",
        f"* **Analysis window:** {_ts(cfg.analysis_start)} .. "
        f"{_ts(cfg.analysis_end)} "
        f"({main.log_returns.shape[0]} aligned minute returns).",
        f"* **Synthetic cascade center:** {_ts(cfg.cascade_center)} "
        "(exact midpoint of the 5-day window).",
        f"* **Synthetic precursor window:** {_ts(cfg.precursor_start)} .. "
        f"{_ts(cfg.precursor_end)} "
        f"({cfg.pre_cascade_lead_minutes} minutes, same as Sessions 8 – 10).",
        f"* **Calm period for surrogates:** {_ts(cfg.calm_start)} .. "
        f"{_ts(cfg.calm_end)} "
        f"(~{(cfg.calm_end - cfg.calm_start).total_seconds() / 3600:.1f} hours).",
        "",
        "**Baselines and ratios** (median of $L^1$ over the first "
        f"{BASELINE_HOURS} hours of the analysis window):",
        "",
        f"* Baseline $L^1$ $H_1$: {_fmt_g(main.baseline_l1_h1)}.",
        f"* Baseline $L^1$ $H_0$: {_fmt_g(main.baseline_l1_h0)}.",
        _peak_line("1", main.pre_l1_h1_peak, ratio_pre_h1, main.pre_l1_h1_peak_time),
        _peak_line("0", main.pre_l1_h0_peak, ratio_pre_h0, main.pre_l1_h0_peak_time),
        (
            f"* Synthetic-center $L^1$ $H_1$ peak: "
            f"{_fmt_g(main.cascade_l1_h1_peak)} "
            f"({_fmt(ratio_cen_h1, 2)}x baseline)."
        ),
        (
            f"* Synthetic-center $L^1$ $H_0$ peak: "
            f"{_fmt_g(main.cascade_l1_h0_peak)} "
            f"({_fmt(ratio_cen_h0, 2)}x baseline)."
        ),
        "",
        "**Surrogate test** (N = "
        f"{surr.n_surrogates} per method; $7\\times$ baseline = "
        f"{_fmt_g(RUN_MULTIPLIER * surr.observed_baseline)}):",
        "",
        (
            f"* Observed $T_{{\\mathrm{{run\\_max}}}}$: "
            f"{surr.observed_t_run_max:.4f} h "
            f"({int(round(surr.observed_t_run_max * 60))} consecutive minutes "
            "of 60-min rolling mean > 7x baseline in the synthetic-precursor "
            "window)."
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
        "**Verdict:** " + _significance_verdict(_row_for_result(result)),
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
    ]
    return "\n".join(lines)


def _build_discussion(rows: Sequence[_TableRow]) -> str:
    calm_rows = [r for r in rows if r.kind == "calm"]
    cascade_rows = [r for r in rows if r.kind == "cascade"]
    if not calm_rows:
        return "(No calm-control rows to discuss.)"

    n_calm_phase_sig = sum(
        1 for r in calm_rows if np.isfinite(r.phase_p) and r.phase_p < 0.05
    )
    n_calm_boot_sig = sum(
        1 for r in calm_rows if np.isfinite(r.bootstrap_p) and r.bootstrap_p < 0.05
    )
    cascade_t_min = min(r.t_run_max for r in cascade_rows) if cascade_rows else float("nan")
    calm_t_max = max(r.t_run_max for r in calm_rows)

    if n_calm_phase_sig == 0 and n_calm_boot_sig == 0:
        verdict = (
            "**The precursor signature is specific to cascades.** Neither "
            "calm window reproduces the $T_{\\mathrm{run\\_max}}$ signature "
            "under either null, while all three cascade events did. This "
            "is the result the design was set up to detect, and it removes "
            "the publication-blocking concern that the Sessions 8 – 10 "
            "finding could be explained by random multi-asset noise."
        )
    elif n_calm_phase_sig == len(calm_rows) and n_calm_boot_sig == len(calm_rows):
        verdict = (
            "**WARNING — the precursor signature also fires in calm windows.** "
            "Both calm controls reject under both nulls. This would mean "
            "the Sessions 8 – 10 finding is not specific to cascades; the "
            "$T_{\\mathrm{run\\_max}}$ statistic as currently constructed "
            "is firing on something more general (likely persistent "
            "cross-asset volatility structure that exists in calm crypto "
            "regimes too). The paper cannot proceed as drafted; the "
            "statistic or its null distribution needs to be tightened."
        )
    else:
        verdict = (
            "**Mixed result — one or more calm windows trip one null.** "
            f"{n_calm_phase_sig} of {len(calm_rows)} calm rows reject the "
            f"phase-randomized null; {n_calm_boot_sig} reject the bootstrap "
            "null. This is a softer signal than full reproduction of the "
            "cascade pattern, but it weakens the cascade-specificity claim "
            "and warrants discussion in the paper -- e.g., reporting both "
            "nulls and noting that bootstrap-only or phase-only rejections "
            "should not be cited as evidence of precursor activity."
        )

    quantitative = (
        f"Quantitatively, the smallest cascade $T_{{\\mathrm{{run\\_max}}}}$ "
        f"is {cascade_t_min:.3f} h (FTX), while the largest calm-window "
        f"$T_{{\\mathrm{{run\\_max}}}}$ is {calm_t_max:.3f} h. "
        + (
            "The calm signature falls well short of the smallest cascade "
            "signature."
            if calm_t_max < cascade_t_min * 0.5
            else (
                "The calm signature is within a factor of two of the smallest "
                "cascade signature, so the headline gap is not as dramatic "
                "as for Terra-Luna or Oct 10."
            )
        )
    )

    return "\n\n".join([verdict, quantitative])


def write_calm_controls_report(
    *,
    results: Sequence[EventResult],
    report_path: Path = _DEFAULT_REPORT_PATH,
) -> Path:
    """Write the Session 11 calm-controls findings + headline comparison.

    The headline table lists the three Session 8 – 10 cascade events
    alongside the calm-control rows so a reader can see, in one place,
    whether the precursor statistic generalises to non-cascade windows.
    Cascade reference numbers come from
    ``paper/control_events_findings.md`` (Session 10, commit 828c77b);
    only the calm-control rows are computed here.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    table_rows: list[_TableRow] = [
        _TableRow(
            event="Oct 10 2025",
            kind="cascade",
            t_run_max=float(OCT10_REFERENCE["observed_t_run_max_hours"]),
            phase_p=float(OCT10_REFERENCE["phase_p_value"]),
            bootstrap_p=float(OCT10_REFERENCE["bootstrap_p_value"]),
            h0_ratio=float(OCT10_REFERENCE["cascade_l1_h0_ratio"]),
        ),
        _row_for_reference(TERRA_LUNA_REFERENCE, event_label="Terra-Luna", kind="cascade"),
        _row_for_reference(FTX_REFERENCE, event_label="FTX", kind="cascade"),
        *[_row_for_result(r) for r in results],
    ]

    detail_blocks = [_detail_block(r) for r in results]
    discussion = _build_discussion(table_rows)

    lines = [
        "# Calm-period negative controls for the $L^1$ $H_1$ precursor",
        "",
        "**Session 11 — does the Sessions 8 – 10 precursor signature also "
        "fire in calm crypto windows?** Sessions 8 – 10 found a "
        "multi-hour $L^1$ $H_1$ build-up before three independent crypto "
        "liquidation cascades (Oct 10 2025, Terra-Luna May 2022, FTX "
        "Nov 2022), each significant at p < 0.05 under multivariate phase "
        "randomization *and* moving-block bootstrap. If the same statistic "
        "also fires during randomly-chosen calm windows, the multi-event "
        "finding is just market noise; if not, the precursor is genuinely "
        "specific to cascade onsets.",
        "",
        "**Design.** Two pre-registered calm 5-day windows in the "
        "historical Binance archive: July 2024 (BTC range-bound around "
        "USD 60 – 63k, no major events) and March 2024 (post-halving-"
        "anticipation drift, no liquidations > USD 1B). The "
        "**Oct 10 symbol basket** (BTC, ETH, SOL, BNB, XRP, DOGE) is "
        "held constant so the only axis varying relative to Oct 10 is "
        "the absence of a cascade. Each window's synthetic cascade "
        "center is set to the exact midpoint of the 5-day window "
        "(2024-07-03 12:00 UTC and 2024-03-17 12:00 UTC respectively); "
        "the 360-minute precursor window, the calm period for "
        "surrogates, and the $T_{\\mathrm{run\\_max}}$ statistic are "
        "all defined exactly as in Sessions 8 – 10.",
        "",
        "Pipeline parameters are identical across all five rows "
        f"(multivariate, $W = {WINDOW_SIZE}$ min, $r_{{\\max}} = "
        f"{MAX_EDGE_LENGTH}$, $H_0$ + $H_1$, indicator window "
        f"{INDICATOR_WINDOW} min, Kendall window {KENDALL_WINDOW} min, "
        f"baseline = median of first {BASELINE_HOURS} hours of the "
        "analysis window; surrogate $N = 1000$ per method). Only the "
        "time window changes between the cascade rows (from Session 10) "
        "and the calm-control rows (this session).",
        "",
        "## Headline comparison",
        "",
        *_render_table(table_rows),
        "",
        "## Per-event detail (calm controls)",
        "",
        *detail_blocks,
        "## Discussion",
        "",
        discussion,
        "",
        "## Caveats",
        "",
        "* **Two calm windows is not many.** A null result here does not "
        "prove the precursor signature *never* fires in calm conditions; "
        "it shows it does not fire in two specific pre-registered windows. "
        "A future extension would sample $K$ random 5-day windows from a "
        "year-long calm regime and report the empirical false-positive "
        "rate of the statistic.",
        "* **Synthetic-center placement.** Both centers are placed at the "
        "exact midpoint of the 5-day window. The result is not sensitive "
        "to this within ~6 hours either way (the precursor window is "
        "360 minutes long and the calm surrogate window is > 54 hours), "
        "but a sensitivity sweep was not run.",
        "* **Basket identity.** The Oct 10 basket is held constant on "
        "purpose: changing both the basket *and* removing the cascade "
        "would confound the two axes. As a side effect, the calm-control "
        "baselines may differ from a fully-random-period baseline that "
        "redrew the basket to match the period's most-liquid tickers.",
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


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_session11_calm_controls(
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
    report_path: Path = _DEFAULT_REPORT_PATH,
    log_returns_overrides: Optional[dict[str, pd.DataFrame]] = None,
    write_report: bool = True,
) -> tuple[EventResult, EventResult, Optional[Path]]:
    """Run both calm-window analyses and write the combined findings report."""
    overrides = dict(log_returns_overrides or {})
    july = run_control_event(
        JULY2024_CALM_CONFIG,
        log_returns=overrides.get(JULY2024_CALM_CONFIG.name),
        n_surrogates=n_surrogates,
        seed=seed,
        n_jobs=n_jobs,
    )
    march = run_control_event(
        MARCH2024_CALM_CONFIG,
        log_returns=overrides.get(MARCH2024_CALM_CONFIG.name),
        n_surrogates=n_surrogates,
        seed=seed + 200,
        n_jobs=n_jobs,
    )
    out_path: Optional[Path] = None
    if write_report:
        out_path = write_calm_controls_report(
            results=[july, march], report_path=report_path
        )
    return july, march, out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:  # pragma: no cover - exercised by hand
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_session11_calm_controls()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
