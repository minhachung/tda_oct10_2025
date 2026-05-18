# Surrogate test of the October 10, 2025 L^1 H_1 precursor

**Session 9 — null-distribution test of the Session 8 precursor finding.** Session 8 reported a 4 – 6 hour pre-cascade build-up in $\|\lambda\|_1^{H_1}$ at minute resolution: peak 16.95x baseline at 2025-10-10 17:14 UTC (241 minutes before the cascade), with hourly means 3 – 7x baseline across the 15:15 – 21:15 UTC window. The open question was whether this elevation is statistically distinguishable from noise, or whether a 4 – 6 hour stretch above 7x baseline can be produced by chance in calm data.

## Setup

* Calm period: 2025-10-08 00:00 UTC .. 2025-10-10 17:00 UTC (~65 hours of pre-precursor data).
* Observed precursor window: 2025-10-10 17:14 UTC .. 2025-10-10 21:14 UTC (strictly before cascade center 21:15 UTC).
* TDA pipeline: `multivariate`, W = 50 min, $r_{\max} = 0.1$, $H_0$ and $H_1$ (Session 8 settings).
* Statistic: $T_{\mathrm{run\_max}}$ = longest run (in fractional hours) of the 60-minute rolling mean of $\|\lambda\|_1^{H_1}$ above $7 \times$ baseline.
* Surrogate baseline: median over first 180 indicator-window samples (= 3 hours).
* Observed baseline: Session 8 value (median over 2025-10-08 00:00 UTC .. 03:00 UTC) = 2.059e-08.
* $7 \times$ observed baseline = 1.441e-07.
* Number of surrogates per method: 1000.
* Symbols: BTC, ETH, SOL, BNB, XRP, DOGE (six majors).

## Observed

* **Observed $T_{\mathrm{run\_max}}$:** 0.7833 hours (47 consecutive minutes of rolling mean above 7x baseline within the 17:14 – 21:14 UTC window).
* Total minutes in the precursor window where the rolling mean exceeds threshold: 47 / 241.

## Surrogate quantiles

| method | $q_{0.05}$ | $q_{0.50}$ | $q_{0.95}$ | max | p-value | verdict |
|---|---|---|---|---|---|---|
| Phase-randomized (Prichard-Theiler multivariate) | 0.000 h | 0.000 h | 0.000 h | 0.000 h | 0.000999 | SIGNIFICANT at p<0.05 |
| Moving-block bootstrap (Kuensch 1989) | 0.000 h | 0.000 h | 0.000 h | 0.300 h | 0.000999 | SIGNIFICANT at p<0.05 |

## Reading the table

* The **phase-randomized null** preserves each column's power spectrum *and* the cross-spectrum between symbols (Prichard & Theiler 1994 shared-phase trick). This is the demanding null for a multivariate TDA pipeline because lagged cross-correlations between BTC/ETH/SOL/BNB/XRP/DOGE are exactly the structure the point cloud sees.
* The **bootstrap null** (moving-block, Politis-White block length rule) preserves the marginal distribution exactly but destroys long-range serial dependence. It is the less strict comparison and reproduces the null used by Ismail et al. (2020).
* iAAFT is *not* used because there is no consensus multivariate formulation that preserves cross-spectra and marginals jointly; applying iAAFT independently per column would destroy the cross-correlations driving the multivariate Vietoris-Rips topology.

## Verdict

Under phase randomization the precursor is **SIGNIFICANT at p<0.05** (empirical p = 0.000999).
Under moving-block bootstrap the precursor is **SIGNIFICANT at p<0.05** (empirical p = 0.000999).

## Caveats

* The conservative North et al. (2002) p-value convention $p = (1 + k) / (1 + N)$ gives a smallest reportable p of $1 / (1 + 1000) = 0.000999$. p-values at this floor mean *no* surrogate equalled or exceeded the observed statistic.
* The 60-minute rolling mean threshold is a *single* exceedance criterion. The Session 8 hourly table shows the per-hour means vary across 3 – 7x baseline within the 15:15 – 21:15 UTC window -- the 7x criterion only triggers in the densest sub-windows. Observed $T_{\mathrm{run\_max}}$ = 0.783 h (47 minutes) is the longest such densely-exceeding stretch within the strict pre-cascade window.
* The surrogate window is ~64 hours; the observed precursor window is ~4 hours. The surrogate has 16x more opportunities to produce a long run, so the test is conservative (biased against rejecting the null).
* The two daily-resolution null tests (Terra-Luna, FTX) have not been re-run at minute resolution yet — that is Session 10.

## Outputs

* Figure: `paper/figures/oct10_surrogate_test.png`
* This report: `paper/oct10_surrogate_findings.md`

