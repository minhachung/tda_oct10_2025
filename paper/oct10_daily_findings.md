# Daily-resolution TDA of the October 10, 2025 crypto liquidation cascade

**Session 7 — first scientific results.** Pipeline calibration and two replication benchmarks (Lorenz Fig. 5a; BTC/ETH/LTC/XRP 2017-2018 Fig. 7c) are documented in `paper/figures/` and `tests/`. This note reports the first novel application of the calibrated pipeline.

## Setup

* Symbols: BTC, ETH, SOL, BNB, XRP, DOGE  (6 majors)
* Window: 2025-08-01 – 2025-11-30 (daily closes, 121 log-return rows)
* TDA: `multivariate`, W=50, $r_{\max}=0.1$, $H_0$ and $H_1$.
* Indicators: AC1, VAR, MPS over a 20-step rolling window of $\|\lambda\|_1^{H_1}$.
* Trend test: rolling Kendall $\tau$ with window 10.
* Baseline reference period: 2025-08-15 – 2025-10-05.
* Data source: CryptoCompare `histoday`. Binance (the canonical Session 5 source via `data_ingestion.load_aligned_returns`) is geo-restricted from this execution environment (HTTP 451); the alignment + log-return recipe is identical, applied to CryptoCompare daily closes.

## Summary table

```
           baseline (Aug 15–Oct 5) 2025-10-03 (week before) 2025-10-10 (cascade) 2025-10-17 (week after)
metric                                                                                                  
L1_H0                     0.004586                 0.004586             0.007979                0.007584
L1_H1                    2.178e-05                2.784e-05            1.832e-05               1.835e-05
L2_H0                      0.01075                  0.01075              0.01527                 0.01483
L2_H1                    0.0001804                 0.000201            0.0001468               0.0001554
PE_H1                        2.802                    3.025                 3.03                   2.785
AC1(L1_H1)                   NaN                      NaN                 0.7674                  0.7895
VAR(L1_H1)                   NaN                      NaN              7.346e-11               7.645e-11
MPS(L1_H1)                   NaN                      NaN                 0.1159                  0.1479
tau_AC1                      NaN                      NaN                  NaN                     NaN  
tau_VAR                      NaN                      NaN                  NaN                     NaN  
tau_MPS                      NaN                      NaN                  NaN                     NaN  
```

## Headline observations

* **$H_1$ landscape norm peaks *after* the cascade, not on it.** $\|\lambda\|_1^{H_1}$ on 2025-10-10 is 1.83e-05 (0.84× the baseline median of 2.18e-05). The maximum over the post-cascade window is 6.05e-05 on 2025-11-04 (2.8× baseline). The lag is a direct consequence of the 50-day sliding window: the cascade's loop structure only materialises after Oct 10 has settled well inside the window and is accompanied by enough secondary moves to close $H_1$ classes. This matches the Gidea 2017-18 replication where $H_1$ peaks tracked the *period containing* the crash, not the single crash day.

* **The $H_0$ landscape norm RISES across the cascade, contradicting the pre-stated hypothesis.** $\|\lambda\|_1^{H_0}$ baseline is 0.00459; on 2025-10-10 it is 0.00798 (1.74× baseline), and the maximum over the post-cascade window is 0.0103 on 2025-11-18 (2.2× baseline). The simple intuition (correlations → 1 ⇒ point cloud compresses ⇒ components merge earlier ⇒ $L^1_{H_0}$ falls) does not hold at daily resolution with a 50-day window: the cascade day is a *single far outlier* in the point cloud and its $H_0$ persistence interval (birth = 0, death = the distance at which the outlier joins the bulk) is therefore *larger*, not smaller. The novelty of looking at $H_0$ holds, but the predicted sign is wrong. Whether the H_0 *drop* hypothesis recovers at minute resolution — where the cascade spans many windows rather than living inside a single one — is the Session 8 question.

* **Kendall $\tau$ on VAR rises sharply *post-cascade*, crossing the Ismail 0.6 threshold.** $\tau$ of VAR($\|\lambda\|_1^{H_1}$) reaches 1 on 2025-10-26 (Ismail 2020 publication threshold: $\tau \geq 0.6$). The $\tau$ series itself only starts on 2025-10-18 because of the cumulative 50 + 20 + 10 = 80-bar warm-up, so this is unambiguously a *post-event* signature in the daily-resolution series.

## Caveats

* The data window is narrow on purpose (122 daily bars). The cumulative warm-up of 50 + 20 + 10 = 80 bars means the Kendall $\tau$ series only starts roughly a week after the cascade; a wider pre-cascade history (Sessions 8 & 9) is required before any claim that the indicator *predicts* the event rather than co-moves with it.
* No surrogate test has been run yet (Session 9).
* No Terra-Luna / FTX control has been computed yet (Session 8).
* No publication-quality polish (Session 11).

## Outputs

* Figure: `paper/figures/oct10_daily_main.png` (4 panels — BTC+ETH prices, $L^1$ $H_1$, $L^1$ $H_0$, Kendall $\tau$ of VAR)
* Processed log-returns: `data/processed/oct10_daily_returns.parquet`

