# Data — sources, layout, and known issues

This directory holds raw and processed market data for the TDA study of
the October 10, 2025 liquidation cascade.

## Layout

```
data/
├── raw/         # Parquet cache for Binance REST responses
│                #   <PAIR>_<kind>_<freq>_<startYYYYMMDD>_<endYYYYMMDD>.parquet
│                #   kind ∈ {spot, futures, funding, oi}
└── processed/   # Aligned/derived artefacts produced downstream
```

Files in `raw/` are written by `tda_oct10.data_ingestion`. The filename
encodes every dimension of the cache key, so the directory listing is
self-documenting and re-runs of `fetch_all()` are idempotent.

## Sources

Two complementary Binance sources are used:

| Source                                  | Used for                    | Notes |
|-----------------------------------------|-----------------------------|-------|
| `api.binance.com` / `fapi.binance.com`  | Spot / perp / funding / OI  | Geo-blocked from this environment (HTTP 451); REST helpers below remain for portability. |
| `data.binance.vision`                   | Bulk historical klines      | Static S3 mirror, **no geo restriction**. Used by Session 8 for minute-resolution data. |
| `min-api.cryptocompare.com/data/v2/histoday` | Daily closes (Session 7) | Free tier, no auth. Used by `validation.fetch_daily_close` and `analysis_oct10.fetch_daily_returns`. |

REST endpoints (`fetch_klines`, `fetch_funding_rate`,
`fetch_open_interest_hist`):

| Endpoint                              | Used for                  | Cadence per request |
|---------------------------------------|---------------------------|---------------------|
| `GET /api/v3/klines`                  | Spot OHLCV                | up to 1000 candles  |
| `GET /fapi/v1/klines`                 | Perp OHLCV (optional)     | up to 1000 candles  |
| `GET /fapi/v1/fundingRate`            | Historical funding rates  | up to 1000 events   |
| `GET /futures/data/openInterestHist`  | Historical open interest  | up to 500 buckets   |

Rate budget: 1200 weight/min. The shared `RateLimiter` is configured
conservatively at 1000 req/min to leave headroom for weight-2 calls.

Bulk archive (`fetch_monthly_klines_archive`,
`load_minute_returns_archive`):

| Path template                                                          | Used for          |
|------------------------------------------------------------------------|-------------------|
| `/data/spot/monthly/klines/{PAIR}/{interval}/{PAIR}-{interval}-YYYY-MM.zip` | Monthly 1m klines |

The archive ZIPs contain a single CSV whose first column is the
open-time. October-2025 files use **microsecond** timestamps in this
column (Binance changed the format mid-2025); earlier months use
milliseconds. `data_ingestion._detect_ts_scale` normalises both into a
UTC `DatetimeIndex` named `"timestamp"`.

## Symbol universe

Six USDT-quoted majors: `BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `DOGE`.
`data_ingestion._to_pair` normalises bare tickers to `<X>USDT` for the
Binance API.

## Time windows

Main study windows (October 10, 2025):

- **Daily**: 2025-08-01 → 2025-11-30
- **Hourly**: 2025-09-01 → 2025-11-15
- **1-minute**: 2025-10-08 00:00 UTC → 2025-10-12 23:59 UTC

Control windows:

- **Terra-Luna collapse**: 2022-05-01 → 2022-05-20 (daily, hourly, 1m)
- **FTX collapse**:        2022-11-01 → 2022-11-20 (daily, hourly, 1m)

All timestamps are UTC throughout the codebase.

## Known data quality issues

### Open interest history retention

`/futures/data/openInterestHist` retains **only ~30 days of history**.
Calls for the Terra-Luna or FTX control windows return an empty frame
— this is expected, not a bug. For these controls, we rely on funding
rates and spot OHLCV alone. If historical OI is required for the
controls, see Binance's downloadable monthly archives at
`https://data.binance.vision/` (not implemented here).

### Funding rate listing dates

USDT-margined perpetuals listed on Binance Futures:

| Symbol  | Listed       |
|---------|--------------|
| BTCUSDT | 2019-09-08   |
| ETHUSDT | 2019-11-27   |
| BNBUSDT | 2020-02-10   |
| XRPUSDT | 2020-01-06   |
| DOGEUSDT| 2020-07-10   |
| SOLUSDT | 2020-09-14   |

All six therefore have funding-rate history covering the 2022 controls
and the 2025 main window.

### Spot kline retention

Spot OHLCV is available back to each symbol's spot listing date (all of
our majors predate 2022). No retention concerns for any of our windows.

### Exchange downtime and gap behaviour

Binance has occasional short maintenance windows (typically <5 min
between candles). `load_aligned_returns` forward-fills runs of missing
bars whose duration is strictly less than 5 minutes, and emits a
`UserWarning` for anything longer so the caller can decide whether to
drop the affected window. Known historical interruptions to flag:

- **2024-12-22**: brief Binance Futures matching engine pause (~3 min).
- **2024-09-04**: spot API HTTP 5xx burst (~2 min).
- **2025-10-10 00:00–06:00 UTC**: extreme volume; no documented outage,
  but expect occasional missing 1-minute candles for the long-tail
  symbols (XRP, DOGE).

If you see large-gap warnings outside these known events, treat them
as suspect and verify against `https://www.binance.com/en/support/announcement`.

### Delisted / renamed symbols

None of the six majors in our universe have been delisted or renamed
across any of the study windows.

## Reproducing the cache

```bash
python -m tda_oct10.data_ingestion
```

This calls `fetch_all()` over every `(window, symbol)` combination. The
first run takes ~10–20 minutes depending on network latency; subsequent
runs are no-ops because the cache is consulted before any HTTP request.
