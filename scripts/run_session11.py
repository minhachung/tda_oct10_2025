"""Driver script for the Session 11 calm-control analysis.

Runs both calm windows end-to-end (with the report writer deferred so
the orchestrator can inspect the surrogate p-values before deciding
whether to commit), dumps key statistics to JSON, and only writes
``paper/calm_controls_findings.md`` once the inspection completes.

Usage::

    PYTHONPATH=src python scripts/run_session11.py

Prints a single-line summary per event so a tail of the log can be
quickly scanned for the headline numbers.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from tda_oct10 import analysis_calm_controls as acc  # noqa: E402


def _result_to_dict(result) -> dict:
    main = result.main
    surr = result.surrogate
    cfg = main.config
    return {
        "name": cfg.name,
        "label": cfg.label,
        "analysis_start": str(cfg.analysis_start),
        "analysis_end": str(cfg.analysis_end),
        "cascade_center": str(cfg.cascade_center),
        "precursor_start": str(cfg.precursor_start),
        "precursor_end": str(cfg.precursor_end),
        "calm_start": str(cfg.calm_start),
        "calm_end": str(cfg.calm_end),
        "n_log_returns": int(main.log_returns.shape[0]),
        "n_landscape_rows": int(main.landscape.shape[0]),
        "baseline_l1_h1": float(main.baseline_l1_h1),
        "baseline_l1_h0": float(main.baseline_l1_h0),
        "pre_l1_h1_peak": float(main.pre_l1_h1_peak),
        "pre_l1_h1_peak_time": str(main.pre_l1_h1_peak_time),
        "pre_l1_h0_peak": float(main.pre_l1_h0_peak),
        "pre_l1_h0_peak_time": str(main.pre_l1_h0_peak_time),
        "cascade_l1_h1_peak": float(main.cascade_l1_h1_peak),
        "cascade_l1_h0_peak": float(main.cascade_l1_h0_peak),
        "observed_t_run_max_hours": float(surr.observed_t_run_max),
        "observed_precursor_above_count": int(surr.observed_precursor_above_count),
        "phase_t_run_max_max": float(np.max(surr.phase_t_run_max)),
        "phase_t_run_max_mean": float(np.mean(surr.phase_t_run_max)),
        "phase_p_value": float(surr.phase_p_value),
        "bootstrap_t_run_max_max": float(np.max(surr.bootstrap_t_run_max)),
        "bootstrap_t_run_max_mean": float(np.mean(surr.bootstrap_t_run_max)),
        "bootstrap_p_value": float(surr.bootstrap_p_value),
        "n_surrogates": int(surr.n_surrogates),
        "data_warnings": list(main.data_warnings),
        "main_plot_path": str(main.main_plot_path) if main.main_plot_path else None,
        "surrogate_plot_path": (
            str(surr.surrogate_plot_path) if surr.surrogate_plot_path else None
        ),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("session11")

    artifacts_dir = ROOT / "data" / "processed"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_json = artifacts_dir / "session11_results.json"

    logger.info("starting Session 11 calm-control analysis (no report yet)")
    t0 = time.time()
    july, march, _ = acc.run_session11_calm_controls(
        n_surrogates=1000,
        seed=42,
        n_jobs=-1,
        write_report=False,
    )
    elapsed = time.time() - t0
    logger.info("both windows complete in %.1f s", elapsed)

    payload = {
        "elapsed_seconds": elapsed,
        "n_surrogates": 1000,
        "results": {
            "july2024": _result_to_dict(july),
            "march2024": _result_to_dict(march),
        },
    }
    results_json.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("results JSON: %s", results_json)

    for tag, result in (("JULY 2024", july), ("MARCH 2024", march)):
        s = result.surrogate
        m = result.main
        logger.info(
            "%s: T_run_max=%.4f h, phase_p=%.4g, bootstrap_p=%.4g, "
            "H1 pre-ratio=%.2fx, H0 cen-ratio=%.2fx",
            tag,
            s.observed_t_run_max,
            s.phase_p_value,
            s.bootstrap_p_value,
            (m.pre_l1_h1_peak / m.baseline_l1_h1) if m.baseline_l1_h1 else float("nan"),
            (m.cascade_l1_h0_peak / m.baseline_l1_h0)
            if m.baseline_l1_h0
            else float("nan"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
