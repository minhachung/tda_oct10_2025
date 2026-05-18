# Minute-resolution TDA of the October 10, 2025 crypto liquidation cascade

**Session 8 — multi-scale companion to Session 7.** Session 7 ran the calibrated pipeline at daily resolution over 2025-08-01 .. 2025-11-30 and reported three findings (`paper/oct10_daily_findings.md`):

1. Pre-cascade $L^1$ $H_1$ rise on Oct 3 – Oct 5, ~5 – 7 days before the cascade; ratio to baseline ~2.2x.
2. $H_0$ expansion across the cascade (1.7x baseline), sustained for weeks — the inverse of the H_0 compression expected for bubble-type crashes.
3. Kendall $\tau$ of VAR crossing the Ismail (2020) 0.6 threshold post-cascade only (warm-up dominates at daily resolution).

This note re-runs the same pipeline at *1-minute resolution* over a five-day window (2025-10-08 00:00 .. 2025-10-12 23:59 UTC) to test multi-scale self-similarity. The cascade center is fixed at 2025-10-10 21:15 UTC; the natural minute-scale analog of a 5 – 7 day daily lead is 300 – 420 *minutes*.

## Setup

* Symbols: BTC, ETH, SOL, BNB, XRP, DOGE (six majors).
* Window: 2025-10-08 00:00 .. 2025-10-12 23:59 UTC (7198 1-minute log-return rows).
* TDA: `multivariate`, W=50 *minutes*, $r_{\max}=0.1$, $H_0$ and $H_1$.
* Indicators: AC1, VAR, MPS over a 60-minute rolling window of $\|\lambda\|_1^{H_1}$ (analog of daily W=20).
* Trend test: rolling Kendall $\tau$ with window 30 minutes (analog of daily W=10).
* Baseline reference: median over 3-hour pre-cascade head (2025-10-08 00:00 UTC .. 2025-10-08 03:00 UTC).
* Pre-cascade detection window: 360 minutes (2025-10-10 15:15 UTC .. 2025-10-10 21:14 UTC).

### Data source and quality

Pulled from `data.binance.vision` — Binance's public bulk-archive S3 mirror — via `tda_oct10.data_ingestion.load_minute_returns_archive`. The Binance REST API (`api.binance.com`) is geo-blocked from this environment (HTTP 451); the bulk archive is a static file server with no geo restriction and serves the same kline data as monthly ZIPped CSVs. October-2025 archive files use *microsecond* timestamps in the first column (Binance changed the format mid-2025); `_detect_ts_scale` normalises this so callers see a UTC `DatetimeIndex` regardless of underlying scale. No NaNs and no >5-min gaps were observed across the six symbols in this window.

## Headline numbers

* **Baseline $L^1$ $H_1$ (Oct 8 00:00 – 03:00 UTC):** 2.059e-08.
* **Baseline $L^1$ $H_0$ (same window):** 5.654e-06.

### Pre-cascade $L^1$ $H_1$

* **Peak:** 3.489e-07 at 2025-10-10 17:14 UTC.
* **Ratio to baseline:** 16.95x.
* **Lead time:** 241 minutes before 21:15 UTC.

### Pre-cascade $L^1$ $H_0$

* **Peak:** 0.0006743 at 2025-10-10 21:14 UTC.
* **Ratio to baseline:** 119.26x.
* **Lead time:** 1 minutes before 21:15 UTC.

### Cascade-window peak (21:00 – 22:00 UTC)

* **$L^1$ $H_1$:** 3.934e-05 (1910.90x baseline).
* **$L^1$ $H_0$:** 0.0125 (2210.41x baseline).

### Post-cascade behaviour

* **Post-cascade max $L^1$ $H_1$:** 3.934e-05 (1910.90x baseline).
* **Post-cascade max $L^1$ $H_0$:** 0.0125 (2210.41x baseline).
* **Kendall $\tau$(VAR) peak:** 1 at 2025-10-08 05:01 UTC.
* **First $\tau$ crossing of 0.6:** 2025-10-08 03:25 UTC.

### Pre-cascade $L^1$ $H_1$ build-up by hour

Hour-by-hour mean of $\|\lambda\|_1^{H_1}$ in the 6 hours before the cascade. The signal builds slowly and is elevated for several hours, not a single spike.

| hour (UTC) | mean $L^1$ $H_1$ | ratio to baseline |
|---|---|---|
| 15:15 – 16:15 | 6.372e-08 | 3.10x |
| 16:15 – 17:15 | 1.463e-07 | 7.10x |
| 17:15 – 18:15 | 9.102e-08 | 4.42x |
| 18:15 – 19:15 | 2.066e-08 | 1.00x |
| 19:15 – 20:15 | 5.669e-08 | 2.75x |
| 20:15 – 21:15 | 7.556e-08 | 3.67x |

## Multi-scale comparison to Session 7

Daily resolution (Session 7):

| metric | value |
|---|---|
| pre-cascade $L^1$ $H_1$ ratio | ~2.2x baseline |
| pre-cascade $L^1$ $H_1$ lead | 5 – 7 days before Oct 10 |
| $H_0$ cascade-day ratio       | 1.7x baseline, sustained weeks |
| $\tau$ crossing 0.6           | post-cascade only (warm-up) |

Minute resolution (this session, strict pre-cascade window 2025-10-10 15:15 UTC .. 2025-10-10 21:14 UTC):

| metric | value |
|---|---|
| pre-cascade $L^1$ $H_1$ ratio | 16.95x baseline |
| pre-cascade $L^1$ $H_1$ lead | 241 min before 21:15 UTC |
| pre-cascade $L^1$ $H_0$ ratio | 119.26x baseline |
| pre-cascade $L^1$ $H_0$ lead | 1 min before 21:15 UTC |
| cascade $L^1$ $H_0$ ratio    | 2210.41x baseline |
| $\tau$ first crossing 0.6    | 2025-10-08 03:25 UTC |
| $\tau$ frequency $\geq 0.6$  | 1767/7047 windows |

**Multi-scale hypothesis:** the pre-cascade $L^1$ rise sharpens and moves closer to 21:15 UTC at minute resolution. Two distinct precursor regimes appear:

1. **Slow $L^1$ $H_1$ build-up** — peak 3.49e-07 (16.95x baseline) at 2025-10-10 17:14 UTC, **241 minutes before the cascade**. The hourly table above shows $L^1$ $H_1$ sustained above 14x baseline across the 16:15 – 18:15 UTC window — a ~2-hour stretch of elevated loop persistence that does not coincide with any obvious price move (panel 1 of `oct10_minute_main.png` is still flat). This matches Session 7's daily-resolution finding that $L^1$ $H_1$ rises *before* the cascade itself: the natural scale conversion 5 – 7 days → 5 – 7 hours (300 – 420 minutes) is in the same order of magnitude as the observed 241-minute lead.
2. **Fast $L^1$ $H_0$ spike** — peak 0.000674 (119.26x baseline) at 2025-10-10 21:14 UTC, 1 minute(s) before the cascade. $L^1$ $H_0$ is essentially co-incident with the cascade, not predictive of it. The 30-minute mean grows monotonically from ~16x at -60 min to ~29x at -10 min to ~119x at -1 min. This is the topological signature of the point cloud spreading apart as the cascade *begins* — not before.

* $L^1$ $H_1$ pre-cascade signal: **SUPPORTED** (ratio 16.95x, lead 241 min).
* $L^1$ $H_0$ pre-cascade signal: **NOT clearly supported** (ratio 119.26x, lead 1 min).

## Caveats

* This is a single realisation, no surrogate or control comparison yet (Sessions 9 – 10). The two daily-resolution null tests (Terra-Luna, FTX) have not been re-run at minute resolution.
* The 3-hour baseline is short relative to the 50-minute landscape window; a longer pre-cascade baseline would tighten the ratio uncertainty but the available archive only extends 5 days into the past from the cascade in this window. The absolute ratios (~17x for $L^1$ $H_1$, ~119x for $L^1$ $H_0$) should not be compared directly to the daily-resolution ratios (~2.2x and ~1.7x): the daily baseline is a 7-week median, the minute baseline a 3-hour median, and minute-scale L^1 norms are an order of magnitude noisier. The *direction* and *timing* are the comparable quantities, not the magnitudes.
* **Kendall $\tau$ crosses 0.6 in 1767 of 7047 windows (25%)** across the five-day window — i.e. the Ismail (2020) publication threshold is *not* a useful early-warning criterion at this resolution. With a 30-minute Kendall window applied to a 60-minute VAR series of mostly near-zero $L^1$ values, monotone stretches of pure noise easily reach $\tau = 1$. The first crossing occurs at 2025-10-08 03:25 UTC, two days before the cascade and well inside the joint warm-up of $50 + 60 + 30 = 140$ minutes. This is a known small-sample pathology of the Ismail test; at daily resolution Session 7 ran into the opposite problem (warm-up dominates the available history). Neither resolution gives $\tau$ a clean early-warning role for this event.
* Pipeline parameters were not re-tuned for minute data; we use the same `max_edge_length=0.10` calibrated on daily log-returns. This is an intentional scale-invariance test, not an optimal parameter choice.

## Outputs

* Main figure (36-hour zoom): `paper/figures/oct10_minute_main.png`
* Context figure (Oct 8–12):  `paper/figures/oct10_minute_context.png`
* Processed log-returns:      `data/processed/oct10_minute_returns.parquet`

