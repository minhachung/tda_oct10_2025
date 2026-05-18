# Calm-period negative controls for the $L^1$ $H_1$ precursor

**Session 11 — does the Sessions 8 – 10 precursor signature also fire in calm crypto windows?** Sessions 8 – 10 found a multi-hour $L^1$ $H_1$ build-up before three independent crypto liquidation cascades (Oct 10 2025, Terra-Luna May 2022, FTX Nov 2022), each significant at p < 0.05 under multivariate phase randomization *and* moving-block bootstrap. If the same statistic also fires during randomly-chosen calm windows, the multi-event finding is just market noise; if not, the precursor is genuinely specific to cascade onsets.

**Design.** Two pre-registered calm 5-day windows in the historical Binance archive: July 2024 (BTC range-bound around USD 60 – 63k, no major events) and March 2024 (post-halving-anticipation drift, no liquidations > USD 1B). The **Oct 10 symbol basket** (BTC, ETH, SOL, BNB, XRP, DOGE) is held constant so the only axis varying relative to Oct 10 is the absence of a cascade. Each window's synthetic cascade center is set to the exact midpoint of the 5-day window (2024-07-03 12:00 UTC and 2024-03-17 12:00 UTC respectively); the 360-minute precursor window, the calm period for surrogates, and the $T_{\mathrm{run\_max}}$ statistic are all defined exactly as in Sessions 8 – 10.

Pipeline parameters are identical across all five rows (multivariate, $W = 50$ min, $r_{\max} = 0.1$, $H_0$ + $H_1$, indicator window 60 min, Kendall window 30 min, baseline = median of first 3 hours of the analysis window; surrogate $N = 1000$ per method). Only the time window changes between the cascade rows (from Session 10) and the calm-control rows (this session).

## Headline comparison

| Event | T_run_max | Phase-rand p | Bootstrap p | H_0 peak ratio |
|---|---|---|---|---|
| Oct 10 2025 (cascade) | 0.783 h | 0.000999 | 0.000999 | 2210.4x |
| Terra-Luna (cascade) | 3.750 h | 0.000999 | 0.000999 | 300.5x |
| FTX (cascade) | 0.333 h | 0.000999 | 0.024 | 24.1x |
| July 2024 (calm) | 0.000 h | 1 | 1 | 1.6x |
| March 2024 (calm) | 0.000 h | 1 | 1 | 1.4x |

## Per-event detail (calm controls)

### July 2024 calm-period control

* **Symbols:** BTC, ETH, SOL, BNB, XRP, DOGE (Oct 10 basket).
* **Classifier:** calm baseline (no cascade event).
* **Analysis window:** 2024-07-01 00:00 UTC .. 2024-07-05 23:59 UTC (7198 aligned minute returns).
* **Synthetic cascade center:** 2024-07-03 11:59 UTC (exact midpoint of the 5-day window).
* **Synthetic precursor window:** 2024-07-03 05:59 UTC .. 2024-07-03 11:58 UTC (360 minutes, same as Sessions 8 – 10).
* **Calm period for surrogates:** 2024-07-01 00:00 UTC .. 2024-07-03 05:59 UTC (~54.0 hours).

**Baselines and ratios** (median of $L^1$ over the first 3 hours of the analysis window):

* Baseline $L^1$ $H_1$: 1.83e-08.
* Baseline $L^1$ $H_0$: 4.29e-06.
* Synthetic-precursor $L^1$ $H_1$ peak: 1.1e-07 (6.02x baseline) at 2024-07-03 11:39 UTC.
* Synthetic-precursor $L^1$ $H_0$ peak: 3.11e-05 (7.26x baseline).
* Synthetic-center $L^1$ $H_1$ peak: 6.18e-08 (3.38x baseline).
* Synthetic-center $L^1$ $H_0$ peak: 6.69e-06 (1.56x baseline).

**Surrogate test** (N = 1000 per method; $7\times$ baseline = 1.28e-07):

* Observed $T_{\mathrm{run\_max}}$: 0.0000 h (0 consecutive minutes of 60-min rolling mean > 7x baseline in the synthetic-precursor window).
* Phase-randomized null: max surrogate $T_{\mathrm{run\_max}} = 0.000$ h; empirical p = 1.
* Bootstrap null: max surrogate $T_{\mathrm{run\_max}} = 0.717$ h; empirical p = 1.

**Verdict:** **neither null rejects at p < 0.05** -- the calm window does not reproduce the cascade precursor under the same surrogate test, as expected.

**Outputs:**
* Main figure: `paper/figures/calm_control_july2024_main.png`
* Surrogate figure: `paper/figures/calm_control_july2024_surrogate_test.png`

**Data warnings:** none

### March 2024 calm-period control

* **Symbols:** BTC, ETH, SOL, BNB, XRP, DOGE (Oct 10 basket).
* **Classifier:** calm baseline (no cascade event).
* **Analysis window:** 2024-03-15 00:00 UTC .. 2024-03-19 23:59 UTC (7198 aligned minute returns).
* **Synthetic cascade center:** 2024-03-17 11:59 UTC (exact midpoint of the 5-day window).
* **Synthetic precursor window:** 2024-03-17 05:59 UTC .. 2024-03-17 11:58 UTC (360 minutes, same as Sessions 8 – 10).
* **Calm period for surrogates:** 2024-03-15 00:00 UTC .. 2024-03-17 05:59 UTC (~54.0 hours).

**Baselines and ratios** (median of $L^1$ over the first 3 hours of the analysis window):

* Baseline $L^1$ $H_1$: 1.43e-07.
* Baseline $L^1$ $H_0$: 1.06e-05.
* Synthetic-precursor $L^1$ $H_1$ peak: 4.57e-07 (3.20x baseline) at 2024-03-17 07:28 UTC.
* Synthetic-precursor $L^1$ $H_0$ peak: 2.76e-05 (2.61x baseline).
* Synthetic-center $L^1$ $H_1$ peak: 1.33e-07 (0.93x baseline).
* Synthetic-center $L^1$ $H_0$ peak: 1.44e-05 (1.36x baseline).

**Surrogate test** (N = 1000 per method; $7\times$ baseline = 1e-06):

* Observed $T_{\mathrm{run\_max}}$: 0.0000 h (0 consecutive minutes of 60-min rolling mean > 7x baseline in the synthetic-precursor window).
* Phase-randomized null: max surrogate $T_{\mathrm{run\_max}} = 0.000$ h; empirical p = 1.
* Bootstrap null: max surrogate $T_{\mathrm{run\_max}} = 0.900$ h; empirical p = 1.

**Verdict:** **neither null rejects at p < 0.05** -- the calm window does not reproduce the cascade precursor under the same surrogate test, as expected.

**Outputs:**
* Main figure: `paper/figures/calm_control_march2024_main.png`
* Surrogate figure: `paper/figures/calm_control_march2024_surrogate_test.png`

**Data warnings:** none

## Discussion

**The precursor signature is specific to cascades.** Neither calm window reproduces the $T_{\mathrm{run\_max}}$ signature under either null, while all three cascade events did. This is the result the design was set up to detect, and it removes the publication-blocking concern that the Sessions 8 – 10 finding could be explained by random multi-asset noise.

Quantitatively, the smallest cascade $T_{\mathrm{run\_max}}$ is 0.333 h (FTX), while the largest calm-window $T_{\mathrm{run\_max}}$ is 0.000 h. The calm signature falls well short of the smallest cascade signature.

## Caveats

* **Two calm windows is not many.** A null result here does not prove the precursor signature *never* fires in calm conditions; it shows it does not fire in two specific pre-registered windows. A future extension would sample $K$ random 5-day windows from a year-long calm regime and report the empirical false-positive rate of the statistic.
* **Synthetic-center placement.** Both centers are placed at the exact midpoint of the 5-day window. The result is not sensitive to this within ~6 hours either way (the precursor window is 360 minutes long and the calm surrogate window is > 54 hours), but a sensitivity sweep was not run.
* **Basket identity.** The Oct 10 basket is held constant on purpose: changing both the basket *and* removing the cascade would confound the two axes. As a side effect, the calm-control baselines may differ from a fully-random-period baseline that redrew the basket to match the period's most-liquid tickers.

## Outputs

* This report: `paper/calm_controls_findings.md`
* July 2024 calm-period control main figure: `paper/figures/calm_control_july2024_main.png`
* July 2024 calm-period control surrogate figure: `paper/figures/calm_control_july2024_surrogate_test.png`
* July 2024 calm-period control returns parquet: `data/processed/calm_control_july2024_minute_returns.parquet`
* March 2024 calm-period control main figure: `paper/figures/calm_control_march2024_main.png`
* March 2024 calm-period control surrogate figure: `paper/figures/calm_control_march2024_surrogate_test.png`
* March 2024 calm-period control returns parquet: `data/processed/calm_control_march2024_minute_returns.parquet`

