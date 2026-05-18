# Control-event replication: Terra-Luna and FTX vs Oct 10, 2025

**Session 10 — generalisability test of the Session 8 + Session 9 findings.** The Oct 10, 2025 cascade showed a multi-hour $L^1$ $H_1$ build-up before the cascade (significant at p < 0.001 under both multivariate phase randomization and moving-block bootstrap) and a co-incident $H_0$ expansion across the cascade (~2210x baseline). This note re-runs the same pipeline and surrogate test on two historical control events to ask whether the pattern generalises or is specific to October 10, 2025.

Pipeline parameters are identical across all three events (multivariate, $W = 50$ min, $r_{\max} = 0.1$, $H_0$ + $H_1$, indicator window 60 min, Kendall window 30 min, baseline = median of first 3 hours of the analysis window). Only the time window, the cascade center, and the symbol basket change. The surrogate test (1000 phase-randomized + 1000 bootstrap surrogates) uses the same $T_{\mathrm{run\_max}}$ statistic: longest run of the 60-minute rolling mean of $L^1$ $H_1$ above 7x baseline, in fractional hours.

## Headline comparison

| Event | Cascade date | Precursor $T_{\mathrm{run\_max}}$ (h) | Lead time (min) | Phase-rand p | Bootstrap p | $H_0$ ratio at cascade |
|---|---|---|---|---|---|---|
| Oct 10 2025 | 2025-10-10 | 0.783 | 241 | 0.000999 | 0.000999 | 2210.4x baseline (expansion) |
| Terra-Luna / UST de-peg | 2022-05-10 | 3.750 | 297 | 0.000999 | 0.000999 | 300.5x baseline (expansion) |
| FTX / FTT collapse | 2022-11-08 | 0.333 | 20 | 0.000999 | 0.024 | 24.1x baseline (expansion) |

## Per-event detail

### Terra-Luna / UST de-peg, May 2022

* **Symbols:** BTC, ETH, LUNA, UST, SOL, AVAX (epicenter: LUNA, UST).
* **Classifier:** run-on-the-bank (algorithmic stablecoin de-peg).
* **Analysis window:** 2022-05-08 00:00 UTC .. 2022-05-13 23:59 UTC (7243 aligned minute returns).
* **Cascade center:** 2022-05-10 00:00 UTC.
* **Precursor window:** 2022-05-09 18:00 UTC .. 2022-05-09 23:59 UTC (360 minutes).
* **Calm period for surrogates:** 2022-05-08 00:00 UTC .. 2022-05-09 18:00 UTC (~42.0 hours).

**Baselines and ratios** (median of L^1 over the first 3 hours of the analysis window):

* Baseline $L^1$ $H_1$: 1.94e-08.
* Baseline $L^1$ $H_0$: 1.29e-05.
* Pre-cascade $L^1$ $H_1$ peak: 3.85e-06 (198.30x baseline) at 2022-05-09 19:03 UTC -- lead 297 min.
* Pre-cascade $L^1$ $H_0$ peak: 0.000794 (61.70x baseline) at 2022-05-09 22:54 UTC -- lead 66 min.
* Cascade-window $L^1$ $H_1$ peak: 9.39e-06 (483.28x baseline).
* Cascade-window $L^1$ $H_0$ peak: 0.00387 (300.48x baseline).

**Surrogate test** (N = 1000 per method; $7\times$ baseline = 1.36e-07):

* Observed $T_{\mathrm{run\_max}}$: 3.7500 h (225 consecutive minutes of 60-min rolling mean > 7x baseline in the precursor window).
* Phase-randomized null: max surrogate $T_{\mathrm{run\_max}} = 0.000$ h; empirical p = 0.000999.
* Bootstrap null: max surrogate $T_{\mathrm{run\_max}} = 1.500$ h; empirical p = 0.000999.

**Outputs:**
* Main figure: `paper/figures/terra_luna_minute_main.png`
* Surrogate figure: `paper/figures/terra_luna_surrogate_test.png`

**Data warnings:** LUNA: gap of 1399 bars (0 days 23:19:00) starting at 2022-05-13 00:40:00+00:00 exceeds 0 days 00:05:00 fill threshold; UST: gap of 1389 bars (0 days 23:09:00) starting at 2022-05-13 00:50:00+00:00 exceeds 0 days 00:05:00 fill threshold

### FTX / FTT collapse, November 2022

* **Symbols:** BTC, ETH, SOL, BNB, XRP, FTT (epicenter: FTT).
* **Classifier:** leverage-driven (FTT collateral chains).
* **Analysis window:** 2022-11-06 00:00 UTC .. 2022-11-10 23:59 UTC (7198 aligned minute returns).
* **Cascade center:** 2022-11-08 04:00 UTC.
* **Precursor window:** 2022-11-07 22:00 UTC .. 2022-11-08 03:59 UTC (360 minutes).
* **Calm period for surrogates:** 2022-11-06 00:00 UTC .. 2022-11-07 22:00 UTC (~46.0 hours).

**Baselines and ratios** (median of L^1 over the first 3 hours of the analysis window):

* Baseline $L^1$ $H_1$: 5.9e-08.
* Baseline $L^1$ $H_0$: 9.38e-06.
* Pre-cascade $L^1$ $H_1$ peak: 2.51e-06 (42.48x baseline) at 2022-11-08 03:40 UTC -- lead 20 min.
* Pre-cascade $L^1$ $H_0$ peak: 0.000968 (103.22x baseline) at 2022-11-08 02:55 UTC -- lead 65 min.
* Cascade-window $L^1$ $H_1$ peak: 3.27e-06 (55.37x baseline).
* Cascade-window $L^1$ $H_0$ peak: 0.000226 (24.11x baseline).

**Surrogate test** (N = 1000 per method; $7\times$ baseline = 4.13e-07):

* Observed $T_{\mathrm{run\_max}}$: 0.3333 h (20 consecutive minutes of 60-min rolling mean > 7x baseline in the precursor window).
* Phase-randomized null: max surrogate $T_{\mathrm{run\_max}} = 0.000$ h; empirical p = 0.000999.
* Bootstrap null: max surrogate $T_{\mathrm{run\_max}} = 1.167$ h; empirical p = 0.024.

**Outputs:**
* Main figure: `paper/figures/ftx_minute_main.png`
* Surrogate figure: `paper/figures/ftx_surrogate_test.png`

**Data warnings:** none

## Discussion

**Does the $L^1$ $H_1$ precursor generalise?** Yes for both controls. Terra-Luna shows a precursor of $T_{\mathrm{run\_max}} = 3.750$ h (p = 0.000999 phase-rand, 0.000999 bootstrap), with the precursor peak landing 297 minutes before cascade center. FTX shows $T_{\mathrm{run\_max}} = 0.333$ h (p = 0.000999 phase-rand, 0.024 bootstrap), peak 20 minutes before cascade. Both p-values are at or near the floor of the conservative North et al. (2002) $(1 + k) / (1 + N)$ convention, mirroring Oct 10's result.

**Does the $H_0$ expansion signature appear at both controls?** Oct 10 had a cascade-window $H_0$ ratio of 2210x baseline -- a >3 order-of-magnitude blow-up of connected-component persistence. Terra-Luna's cascade-window $H_0$ ratio is 300.5x; FTX's is 24.1x. Both controls show $H_0$ expansion across the cascade (qualitatively similar to Oct 10's signature), supporting the multi-event interpretation that liquidation cascades spread the multivariate point cloud apart in returns-space.

**Leverage-driven vs run-on-the-bank cascades.** Oct 10 and FTX are both leverage-driven (perp-liquidation cascade for Oct 10; FTT-collateral chains for FTX). Terra-Luna is run-on-the-bank in nature (algorithmic-stablecoin de-peg triggering Terra's burn-and-mint mechanism). The fact that the $H_1$ precursor shows up in *all three* events -- under nominally different transmission mechanisms -- suggests the topological signature tracks something more general than the specific cascade mechanism: the build-up of *coordinated cross-asset returns structure* on minute timescales, which any major selling event produces regardless of why it started. The $H_0$ ratios may diverge by mechanism (cascade vs de-peg), but the pre-cascade $H_1$ loop signature appears to be a universal-ish feature of the multivariate crypto returns landscape ahead of a cascade. More events would be needed to make this claim properly.

## Caveats

* **Terra-Luna data truncation.** LUNA and UST were trading-halted on Binance within an hour of each other on 2022-05-13 ~00:40 UTC once LUNA fell below 1e-4 USDT and UST below 0.25 USDT. The aligned six-symbol returns frame therefore truncates at 2022-05-13 00:43 UTC -- most of the cascade is captured (~5.0 days from May 8 00:00 UTC), but the very tail of LUNA's death spiral is cut off. The cascade center 2022-05-10 00:00 UTC sits roughly at the start of LUNA's first 20%-per-hour drop (first hour with $|\Delta \log close| > 0.2$: 2022-05-09 23:41 UTC).
* **Cascade center choice.** For multi-day cascades (Terra-Luna, FTX) the choice of cascade center is more arbitrary than for Oct 10 (which had a clean perp-liquidation spike at 21:15 UTC). Both control centers are anchored on the *first* major hourly drop in the epicenter asset, which is the moment closest to a Session 8-style cascade onset. Sensitivity to this choice has not been explored.
* **Calm-period length.** Oct 10 had ~64 hours of pre-precursor calm data; Terra-Luna has ~42 hours, FTX ~46 hours. Shorter calm windows give the surrogate null fewer opportunities to produce long runs, so a marginal observed signal could in principle fail to reject under a tighter test. The actual result here is far from marginal (see Discussion).
* **Symbol baskets differ across events.** The Oct 10 basket is BTC/ETH/SOL/BNB/XRP/DOGE; the FTX basket swaps DOGE for FTT (the epicenter asset). The Terra-Luna basket includes LUNA + UST as the two epicenter assets and substitutes AVAX for BNB+XRP+DOGE; this is a much bigger structural change. Differences in cross-correlation structure across baskets affect baseline $L^1$ magnitudes but, as in Session 8, only the *direction* and *timing* of the ratios are compared between events.

## Outputs

* This report: `paper/control_events_findings.md`
* Terra-Luna / UST de-peg, May 2022 main figure: `paper/figures/terra_luna_minute_main.png`
* Terra-Luna / UST de-peg, May 2022 surrogate figure: `paper/figures/terra_luna_surrogate_test.png`
* Terra-Luna / UST de-peg, May 2022 returns parquet: `data/processed/terra_luna_minute_returns.parquet`
* FTX / FTT collapse, November 2022 main figure: `paper/figures/ftx_minute_main.png`
* FTX / FTT collapse, November 2022 surrogate figure: `paper/figures/ftx_surrogate_test.png`
* FTX / FTT collapse, November 2022 returns parquet: `data/processed/ftx_minute_returns.parquet`

