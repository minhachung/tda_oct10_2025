"""Render ``paper/calm_controls_findings.md`` from a previously-dumped
``data/processed/session11_results.json``.

Decoupled from the analysis runner so the surrogate test can be inspected
before the report is committed. Reconstitutes lightweight stand-ins for
``EventMainResult`` / ``EventSurrogateResult`` from the JSON dump
(everything the writer actually reads) and calls
``analysis_calm_controls.write_calm_controls_report``.

Usage::

    PYTHONPATH=src python scripts/render_session11_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tda_oct10 import analysis_calm_controls as acc  # noqa: E402

_NAME_TO_CONFIG = {
    acc.JULY2024_CALM_CONFIG.name: acc.JULY2024_CALM_CONFIG,
    acc.MARCH2024_CALM_CONFIG.name: acc.MARCH2024_CALM_CONFIG,
}


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _build_mock_result(payload: dict[str, Any]) -> SimpleNamespace:
    """Reconstruct a minimal ``EventResult``-like object from the JSON dump.

    Populates only the attributes the report writer reads. Surrogate arrays
    are reconstructed as length-1 numpy arrays containing the per-method max
    (the writer only calls ``.max()`` on them).
    """
    cfg = _NAME_TO_CONFIG[payload["name"]]
    log_returns_shape = SimpleNamespace(shape=(payload["n_log_returns"], len(cfg.symbols)))
    main = SimpleNamespace(
        config=cfg,
        log_returns=log_returns_shape,
        baseline_l1_h1=payload["baseline_l1_h1"],
        baseline_l1_h0=payload["baseline_l1_h0"],
        pre_l1_h1_peak=payload["pre_l1_h1_peak"],
        pre_l1_h1_peak_time=_ts(payload["pre_l1_h1_peak_time"]),
        pre_l1_h0_peak=payload["pre_l1_h0_peak"],
        pre_l1_h0_peak_time=(
            _ts(payload["pre_l1_h0_peak_time"])
            if payload.get("pre_l1_h0_peak_time")
            and payload["pre_l1_h0_peak_time"] != "None"
            else None
        ),
        cascade_l1_h1_peak=payload["cascade_l1_h1_peak"],
        cascade_l1_h0_peak=payload["cascade_l1_h0_peak"],
        data_warnings=tuple(payload.get("data_warnings", ())),
        main_plot_path=(
            Path(payload["main_plot_path"]) if payload.get("main_plot_path") else None
        ),
    )
    surrogate = SimpleNamespace(
        config=cfg,
        observed_baseline=payload["baseline_l1_h1"],
        observed_t_run_max=payload["observed_t_run_max_hours"],
        observed_precursor_above_count=payload["observed_precursor_above_count"],
        n_surrogates=payload["n_surrogates"],
        phase_t_run_max=np.array([payload["phase_t_run_max_max"]]),
        phase_p_value=payload["phase_p_value"],
        bootstrap_t_run_max=np.array([payload["bootstrap_t_run_max_max"]]),
        bootstrap_p_value=payload["bootstrap_p_value"],
        surrogate_plot_path=(
            Path(payload["surrogate_plot_path"])
            if payload.get("surrogate_plot_path")
            else None
        ),
    )
    return SimpleNamespace(main=main, surrogate=surrogate)


def main() -> int:
    json_path = ROOT / "data" / "processed" / "session11_results.json"
    if not json_path.exists():
        print(f"missing: {json_path}", file=sys.stderr)
        return 1
    payload = json.loads(json_path.read_text())
    july = _build_mock_result(payload["results"]["july2024"])
    march = _build_mock_result(payload["results"]["march2024"])
    out = acc.write_calm_controls_report(results=[july, march])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
