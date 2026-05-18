# Project state — end of analysis phase

External-memory snapshot for the **TDA of the October 10, 2025 cryptocurrency
liquidation cascade** project. Captures everything the paper-writing phase
needs to pick up in a fresh chat: commits, figures, findings docs, source
modules, test inventory, and the master results table.

Snapshot taken at commit `e47acc3` (Session 11), 2026-05-17. Working tree
clean; pushed to `origin/main` at
[github.com/minhachung/tda_oct10_2025](https://github.com/minhachung/tda_oct10_2025).

The analysis phase is complete. **No further analysis sessions.** The next
phase is paper writing, planned separately.

---

## Headline result

The minute-resolution $L^1$ $H_1$ multivariate-TDA precursor signature fires
at three independent crypto liquidation cascades (Oct 10 2025, Terra-Luna
May 2022, FTX Nov 2022) at p < 0.05 under both phase-randomized and
moving-block-bootstrap nulls, **and does not fire** at two pre-registered
calm-period negative controls (July 2024, March 2024). The pre-cascade
$H_1$ build-up and the cascade-window $H_0$ expansion both appear specific
to cascades.

### Master comparison table (5 events)

| Event | Window | Basket | T_run_max | Phase-rand p | Bootstrap p | Pre-cascade H_1 ratio | Lead (min) | H_0 cascade-window ratio |
|---|---|---|---|---|---|---|---|---|
| Oct 10 2025 (cascade) | 2025-10-08 – 2025-10-12 | BTC/ETH/SOL/BNB/XRP/DOGE | 0.783 h | 0.000999 | 0.000999 | 16.95x | 241 | 2210.4x |
| Terra-Luna (cascade) | 2022-05-08 – 2022-05-13 | BTC/ETH/LUNA/UST/SOL/AVAX | 3.750 h | 0.000999 | 0.000999 | 198.30x | 297 | 300.5x |
| FTX (cascade) | 2022-11-06 – 2022-11-10 | BTC/ETH/SOL/BNB/XRP/FTT | 0.333 h | 0.000999 | 0.024 | 42.48x | 20 | 24.1x |
| July 2024 (calm) | 2024-07-01 – 2024-07-05 | BTC/ETH/SOL/BNB/XRP/DOGE | 0.000 h | 1.000 | 1.000 | 6.02x | n/a | 1.6x |
| March 2024 (calm) | 2024-03-15 – 2024-03-19 | BTC/ETH/SOL/BNB/XRP/DOGE | 0.000 h | 1.000 | 1.000 | 3.20x | n/a | 1.4x |

Notes:
- All p-values use the conservative North et al. (2002) $(1 + k) / (1 + N)$
  empirical-p convention with $N = 1000$ surrogates per method, so
  $0.000999 = 1 / 1001$ is the floor.
- "T_run_max" is the longest run (in fractional hours) of the 60-min rolling
  mean of $L^1$ $H_1$ above $7\times$ baseline in the 360-min precursor
  window. Baseline = median of $L^1$ $H_1$ over the first 3 hours of each
  event's analysis window.
- "Pre-cascade H_1 ratio" is the maximum of $L^1$ $H_1$ in the precursor
  window divided by the baseline. "Lead" is the offset of that peak from
  the cascade center.
- "H_0 cascade-window ratio" is the maximum of $L^1$ $H_0$ in
  $[\textrm{center} - 15\,\textrm{min},\ \textrm{center} + 45\,\textrm{min}]$
  divided by the baseline $L^1$ $H_0$.
- For the calm controls there is no cascade; the synthetic cascade center
  is the exact midpoint of each 5-day window (2024-07-03 12:00 UTC and
  2024-03-17 12:00 UTC).

---

## Commit history

```
e47acc3 Session 11: calm-period negative controls for precursor signature
828c77b Session 10: Terra-Luna and FTX control events -- minute-resolution TDA and surrogate tests
42fecbe Session 9: phase-randomized and bootstrap surrogate tests of Oct 10 L^1_H_1 precursor
f55a44f Session 8: minute-resolution TDA analysis of October 10, 2025 cascade
bae4293 docs: reframe Oct 10 daily findings — pre-cascade L1_H1 signal, H0 expansion as cascade signature
683c86f Session 7: daily-resolution TDA analysis of October 10, 2025 cryptocurrency liquidation cascade
3ea9a76 docs: document max_edge_length=0.10 calibration rationale
7306b2b Session 6: Gidea 2020 replication; calibrate max_edge_length=0.10 via parameter sweep on BTC/ETH/LTC/XRP 2016-2018
f8c6a6e Session 5: Binance + CryptoCompare data ingestion with parquet caching
ab2d0f6 Session 4: bootstrap and phase-randomized surrogate distributions
98a7f87 Session 3: indicators (AC1, VAR, MPS) and Kendall tau trend test
4c0d48e Session 2: regression tests for TDAPipeline against Lorenz reference
5d3ef30 test: reproduce Gidea et al. (2020) Fig 5(a) on noisy Lorenz-type map
3ada698 chore: scaffold tda_oct10_2025 project
```

Reading order for a fresh chat: Sessions 5 (data) → 6 (calibration) → 7
(daily) → 8 (minute) → 9 (surrogates) → 10 (control events) → 11 (calm
controls). Sessions 1–4 are infrastructure (scaffold, pipeline tests,
indicators, surrogates).

---

## Figures (`paper/figures/`)

| File | Description |
|---|---|
| `gidea2020_epsilon_sweep.png` | Session 6 parameter sweep of $r_{\max}$ over BTC/ETH/LTC/XRP 2016 – 2018 used to calibrate the pipeline at $r_{\max} = 0.10$. |
| `replication_gidea2020_btc.png` | Session 6 replication of Gidea et al. (2020) Fig. 7(c): rolling $L^1$ $H_1$ on BTC 2017 – 2018 showing the pre-crash build-up. |
| `oct10_daily_main.png` | Session 7 daily-resolution figure for Oct 10 2025: price, $L^1$ $H_1$, $L^1$ $H_0$, Kendall $\tau$ over 2025-08-01 – 2025-11-30. |
| `oct10_minute_main.png` | Session 8 main four-panel minute-resolution figure for Oct 10 2025 (5-day window). |
| `oct10_minute_context.png` | Session 8 context panel — wider Oct 8 – 12 view around the cascade. |
| `oct10_surrogate_test.png` | Session 9 surrogate-test figure for Oct 10: phase-rand + bootstrap null histograms + L^1 H_1 overlay. |
| `terra_luna_minute_main.png` | Session 10 main four-panel for Terra-Luna / UST de-peg (May 2022). |
| `terra_luna_surrogate_test.png` | Session 10 surrogate-test figure for Terra-Luna. |
| `ftx_minute_main.png` | Session 10 main four-panel for FTX / FTT collapse (Nov 2022). |
| `ftx_surrogate_test.png` | Session 10 surrogate-test figure for FTX. |
| `calm_control_july2024_main.png` | Session 11 main four-panel for July 2024 calm-period control (no cascade). |
| `calm_control_july2024_surrogate_test.png` | Session 11 surrogate-test figure for July 2024 (both nulls dominated at 0; observed T = 0). |
| `calm_control_march2024_main.png` | Session 11 main four-panel for March 2024 calm-period control. |
| `calm_control_march2024_surrogate_test.png` | Session 11 surrogate-test figure for March 2024. |

---

## Findings documents (`paper/`)

| File | Description |
|---|---|
| `oct10_daily_findings.md` | Session 7 first scientific result. Daily-resolution pipeline over 2025-08 – 2025-11 reports a multi-day pre-cascade $L^1$ $H_1$ build-up and an $H_0$ expansion across the cascade. |
| `oct10_minute_findings.md` | Session 8 minute-resolution companion to Session 7. Pre-cascade $L^1$ $H_1$ peak 16.95x baseline at 4 h 1 min lead; $H_0$ ratio 2210x at the cascade. |
| `oct10_surrogate_findings.md` | Session 9 null-distribution test of the Session 8 precursor: phase-rand p = 0.000999, bootstrap p = 0.000999 against $T_{\mathrm{run\_max}}$ statistic. |
| `control_events_findings.md` | Session 10 replication on Terra-Luna and FTX. Both cascade events reproduce the precursor signature (p ≤ 0.024 across all four null tests). |
| `calm_controls_findings.md` | Session 11 negative-control test on July 2024 and March 2024 calm windows. Both calm windows produce T_run_max = 0.000 h, p = 1.000 under both nulls — precursor signature is specific to cascades. |

---

## Source modules (`src/tda_oct10/`)

| Module | Description |
|---|---|
| `__init__.py` | Package marker; no runtime logic. |
| `tda_pipeline.py` | `TDAPipeline` — windowed Vietoris-Rips → persistence landscape features ($L^1$/$L^2$ norms for $H_0$ / $H_1$, persistence entropy). Multivariate and per-symbol modes. |
| `data_ingestion.py` | Binance REST + bulk-archive ingestion (`fetch_klines`, `fetch_monthly_klines_archive`, `load_minute_returns_archive`) + CryptoCompare daily fallback, with parquet caching under `data/raw/`. |
| `indicators.py` | Early-warning-signal indicators (AC1, VAR, MPS) following Ismail et al. (2020), computed in rolling windows over a 1-D series. |
| `trend_test.py` | Rolling Kendall-$\tau$ trend test with permutation p-values, applied to AC1/VAR/MPS series. |
| `nulls.py` | Surrogate generators: multivariate phase randomization (Prichard-Theiler) and moving-block bootstrap (Künsch), plus empirical p-value helpers. |
| `validation.py` | Validation utilities — daily-close fetcher for cross-checking against CryptoCompare and Binance daily data. |
| `analysis_oct10.py` | Session 7 driver — daily-resolution pipeline over Aug – Nov 2025, calls `tda_pipeline` + `indicators` + `trend_test`, produces `oct10_daily_main.png` and `oct10_daily_findings.md`. |
| `analysis_oct10_minute.py` | Session 8 driver — minute-resolution pipeline over the 5-day window around 2025-10-10, defines `WINDOW_SIZE=50`, `MAX_EDGE_LENGTH=0.10`, `INDICATOR_WINDOW=60`, `KENDALL_WINDOW=30`, `ISMAIL_TAU_THRESHOLD=0.6`. |
| `analysis_oct10_surrogate.py` | Session 9 driver — phase-rand + bootstrap surrogate test on the Session 8 calm-slice, defines `DEFAULT_N_SURROGATES=1000`, `RUN_MULTIPLIER=7.0`, `SURROGATE_BASELINE_HEAD_MINUTES=180`, $T_{\mathrm{run\_max}}$ statistic. |
| `analysis_control_events.py` | Session 10 driver — parameterised on `EventConfig`, runs the full Session 8 + 9 pipeline on Terra-Luna and FTX. Provides `EventConfig`, `TERRA_LUNA_CONFIG`, `FTX_CONFIG`, `run_control_event`, `write_control_events_report`, and the cascade reference numbers (`OCT10_REFERENCE`). Session 11 extends `EventConfig` with optional `main_plot_filename`, `surrogate_plot_filename`, `center_label`. |
| `analysis_calm_controls.py` | Session 11 driver — `JULY2024_CALM_CONFIG`, `MARCH2024_CALM_CONFIG`, `run_session11_calm_controls`, `write_calm_controls_report`. Holds the Oct 10 basket constant and places the synthetic cascade center at the exact midpoint of each 5-day window. |

### Reproducibility scripts (`scripts/`)

| Script | Description |
|---|---|
| `run_session11.py` | Runs the two calm-window analyses end-to-end, dumps summary statistics to `data/processed/session11_results.json` *before* writing the report so results can be inspected first. |
| `render_session11_report.py` | Reconstructs lightweight `EventResult` stand-ins from the JSON dump and renders `paper/calm_controls_findings.md` without re-running the 85-min surrogate analysis. |

---

## Tests (`tests/`) — 68 / 68 passing at `e47acc3`

| File | Tests | Description |
|---|---:|---|
| `test_tda_pipeline.py` | 2 | Regression tests for the windowed Vietoris-Rips → persistence-landscape pipeline. |
| `test_lorenz.py` | 1 | Reproduction of Gidea et al. (2020) Fig. 5(a) on noisy Lorenz-type map — sanity test for the TDA pipeline against a published reference. |
| `test_data_ingestion.py` | 11 | Klines fetcher, archive fetcher, cache round-trip, rate limiter, gap detection, and CSV-header / timestamp-scale heuristics. |
| `test_indicators_and_trend.py` | 12 | AC1/VAR/MPS rolling indicators and Kendall-$\tau$ trend test with permutation p-values. |
| `test_nulls.py` | 23 | Phase-randomization and moving-block-bootstrap correctness (mean / variance / spectral preservation) plus empirical p-value edge cases. |
| `test_analysis_oct10.py` | 3 | Session 7 daily-resolution driver end-to-end on synthetic returns. |
| `test_analysis_oct10_minute.py` | 3 | Session 8 minute-resolution driver end-to-end on synthetic minute returns with injected cascade-center shock. |
| `test_session9_surrogates.py` | 6 | Session 9 surrogate-test driver — phase-rand + bootstrap null distributions, $T_{\mathrm{run\_max}}$ statistic, empirical-p floor. |
| `test_session10_controls.py` | 5 | Session 10 control-event driver — `EventConfig` self-consistency, Terra-Luna + FTX synthetic end-to-end, combined-report rendering. |
| `test_session11_calm_controls.py` | 2 | Session 11 calm-control driver — both calm `EventConfig`s self-consistent (Oct 10 basket, no epicenter, exact-midpoint synthetic center, brief-mandated filenames), and the combined report renders all five rows. |

Full-suite runtime ≈ 7 min on this machine.

---

## Pipeline parameters (shared across Sessions 8 – 11)

These are the constants the paper will quote; they are defined in
`analysis_oct10_minute.py` and `analysis_oct10_surrogate.py` and reused
unmodified for the control events and calm controls.

| Constant | Value | Meaning |
|---|---|---|
| `WINDOW_SIZE` | 50 min | Sliding window length fed to Vietoris-Rips. |
| `MAX_EDGE_LENGTH` | 0.10 | $r_{\max}$ in normalised-returns units; calibrated in Session 6. |
| `HOMOLOGY_DIMS` | $[0, 1]$ | Homology dimensions retained from each diagram. |
| `INDICATOR_WINDOW` | 60 min | Rolling window for AC1/VAR/MPS indicators and the 60-min rolling mean of $L^1$ $H_1$. |
| `KENDALL_WINDOW` | 30 min | Rolling window for the Kendall-$\tau$ trend test on indicator series. |
| `ISMAIL_TAU_THRESHOLD` | 0.6 | Threshold for "trend present" following Ismail et al. (2020). |
| `BASELINE_HOURS` | 3 | First-3-hours-of-window baseline for $L^1$ ratios. |
| `PRE_CASCADE_LEAD_MINUTES` | 360 | Length of the precursor window before cascade center. |
| `RUN_MULTIPLIER` | 7.0 | $L^1$ $H_1$ runs above $7\times$ baseline define the $T_{\mathrm{run\_max}}$ statistic. |
| `SURROGATE_BASELINE_HEAD_MINUTES` | 180 | Head of the calm slice used as the surrogate-side baseline. |
| `DEFAULT_N_SURROGATES` | 1000 | Per-method surrogate count for both nulls. |
| `DEFAULT_SEED` | 42 | RNG seed for surrogate generation. |

---

## Data sources

- **`data.binance.vision`** — bulk monthly 1-minute kline ZIP archives, no
  geo restriction. Used for Sessions 8 – 11. Cached as parquet under
  `data/raw/<PAIR>_archive_1m_YYYYMM.parquet`.
- **`min-api.cryptocompare.com/data/v2/histoday`** — daily closes for
  cross-validation and the Session 7 daily-resolution pipeline.
- Binance REST endpoints (`api.binance.com`, `fapi.binance.com`) are
  geo-blocked from this environment (HTTP 451). The REST fetchers exist in
  `data_ingestion.py` for portability but are not on the analysis hot
  path.

Raw and processed parquet files are gitignored (`data/raw/`,
`data/processed/`); they re-materialise on first call to
`load_minute_returns_archive` / `fetch_event_returns`.

---

## What goes in the paper

Each session's findings doc is already drafted in `paper/*.md` and the
figures are publication-ready. The paper-writing phase should:

1. Read each `*_findings.md` for the section's narrative + numbers.
2. Use the master table above as the headline empirical result.
3. Pull pipeline parameters from the table above for the methods section.
4. Cite Gidea et al. (2020) for the calibration / replication and Ismail
   et al. (2020) for the Kendall-$\tau$ threshold; cite Prichard & Theiler
   (1994) for multivariate phase randomization and Künsch (1989) for the
   moving-block bootstrap; cite North et al. (2002) for the
   $(1 + k) / (1 + N)$ empirical-p convention.
5. The cleanest structure is probably: methods → calibration replication
   (Session 6) → daily Oct 10 (Session 7, brief) → minute Oct 10
   (Session 8) → surrogates (Session 9) → control events (Session 10) →
   negative controls (Session 11) → discussion.

Open caveats already documented in the findings docs:
- Terra-Luna data truncates 2022-05-13 ~00:43 UTC (LUNA/UST delisting).
- Two calm windows is not many; a future extension would sample $K$
  random 5-day calm windows and report an empirical false-positive rate.
- Cascade-center placement for Terra and FTX is anchored on first major
  hourly drop in the epicenter asset; sensitivity not explored.
- Symbol baskets differ across the three cascade events (DOGE ↔ FTT for
  FTX; AVAX + LUNA + UST replace BNB/XRP/DOGE for Terra).
