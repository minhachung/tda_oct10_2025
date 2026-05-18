"""Data ingestion for the October 10, 2025 liquidation cascade study.

Fetches Binance spot OHLCV, futures funding rates, and historical open
interest for a fixed basket of six majors (BTC, ETH, SOL, BNB, XRP,
DOGE), persists everything to ``data/raw/`` as Parquet, and exposes a
``load_aligned_returns`` helper that returns timestamp-aligned
log-returns suitable for the TDA pipeline.

All endpoints used are public REST; no authentication is required.

The fetchers respect Binance's documented 1200 requests/minute weight
budget via a simple token-bucket rate limiter (see ``RateLimiter``).
Every paginated wrapper streams batches until the requested
``[start, end]`` window is exhausted, with bounded retry on transient
failures.

Caching
-------
Every paginated wrapper consults ``data/raw/`` before issuing any HTTP
request. Cache keys are the tuple ``(symbol, kind, frequency, start,
end)`` and are encoded directly into the filename so that re-runs hit
the cache without any side effects. The HTTP client is the single
function ``_http_get``; tests monkey-patch it to assert that the second
call never reaches the network.

Known data quality issues (gaps, exchange downtime, restricted
endpoints) are documented in ``data/README.md``.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

__all__ = [
    "SYMBOLS",
    "WINDOWS",
    "RateLimiter",
    "fetch_klines",
    "fetch_funding_rate",
    "fetch_open_interest_hist",
    "fetch_all",
    "load_aligned_returns",
    "cache_path",
    "fetch_monthly_klines_archive",
    "load_minute_returns_archive",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS: tuple[str, ...] = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE")

SPOT_BASE = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"

# Binance documented limits.
REQUEST_WEIGHT_PER_MIN = 1200
SAFE_REQUESTS_PER_MIN = 1000  # conservative margin

MAX_KLINES_PER_REQ = 1000
MAX_FUNDING_PER_REQ = 1000
MAX_OI_HIST_PER_REQ = 500

# Interval → milliseconds.
_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

# Default cache root: ``<repo>/data/raw``. Resolved lazily so tests can
# point it elsewhere via ``set_cache_root``.
_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
_CACHE_ROOT: Path = _DEFAULT_CACHE_ROOT


def set_cache_root(path: Path) -> None:
    """Override the directory used for Parquet caches (for tests)."""
    global _CACHE_ROOT
    _CACHE_ROOT = Path(path)
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Window:
    """Inclusive ``[start, end]`` window at a single sampling frequency."""

    name: str
    start: datetime
    end: datetime
    interval: str


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


WINDOWS: dict[str, Window] = {
    # Main Oct 10, 2025 study windows.
    "daily_main": Window("daily_main", _utc(2025, 8, 1), _utc(2025, 11, 30), "1d"),
    "hourly_main": Window("hourly_main", _utc(2025, 9, 1), _utc(2025, 11, 15), "1h"),
    "minute_main": Window(
        "minute_main", _utc(2025, 10, 8), _utc(2025, 10, 12, 23, 59), "1m"
    ),
    # Terra-Luna control.
    "daily_terra": Window("daily_terra", _utc(2022, 5, 1), _utc(2022, 5, 20), "1d"),
    "hourly_terra": Window("hourly_terra", _utc(2022, 5, 1), _utc(2022, 5, 20), "1h"),
    "minute_terra": Window(
        "minute_terra", _utc(2022, 5, 1), _utc(2022, 5, 20, 23, 59), "1m"
    ),
    # FTX collapse control.
    "daily_ftx": Window("daily_ftx", _utc(2022, 11, 1), _utc(2022, 11, 20), "1d"),
    "hourly_ftx": Window("hourly_ftx", _utc(2022, 11, 1), _utc(2022, 11, 20), "1h"),
    "minute_ftx": Window(
        "minute_ftx", _utc(2022, 11, 1), _utc(2022, 11, 20, 23, 59), "1m"
    ),
}


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter.

    Tracks request timestamps for the last 60 seconds and sleeps just
    long enough to keep throughput at or below ``max_per_minute``. Safe
    to share across pagination loops.
    """

    def __init__(self, max_per_minute: int = SAFE_REQUESTS_PER_MIN) -> None:
        if max_per_minute <= 0:
            raise ValueError("max_per_minute must be positive")
        self._max = max_per_minute
        self._window: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        while self._window and self._window[0] < cutoff:
            self._window.popleft()
        if len(self._window) >= self._max:
            sleep_for = 60.0 - (now - self._window[0]) + 0.01
            if sleep_for > 0:
                logger.info("rate-limit sleeping for %.2fs", sleep_for)
                time.sleep(sleep_for)
            now = time.monotonic()
            cutoff = now - 60.0
            while self._window and self._window[0] < cutoff:
                self._window.popleft()
        self._window.append(now)


_DEFAULT_LIMITER = RateLimiter()


# ---------------------------------------------------------------------------
# HTTP layer (single chokepoint — tests monkeypatch this)
# ---------------------------------------------------------------------------


def _http_get(
    url: str,
    params: dict,
    *,
    limiter: RateLimiter = _DEFAULT_LIMITER,
    max_retries: int = 5,
    timeout: float = 30.0,
) -> list | dict:
    """GET ``url`` with rate limiting and bounded exponential backoff.

    Retries on connection errors and on HTTP 418/429/5xx. Raises after
    ``max_retries`` consecutive failures.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            backoff = min(2 ** attempt, 30)
            logger.warning("network error %s; retrying in %ds", exc, backoff)
            time.sleep(backoff)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (418, 429) or resp.status_code >= 500:
            backoff = min(2 ** attempt, 60)
            logger.warning(
                "HTTP %s on %s; retrying in %ds", resp.status_code, url, backoff
            )
            time.sleep(backoff)
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            continue
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    raise RuntimeError(f"giving up after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Symbol + filename helpers
# ---------------------------------------------------------------------------


def _to_pair(symbol: str) -> str:
    """Normalise ``BTC`` → ``BTCUSDT``; pass through if already a pair."""
    s = symbol.upper()
    return s if s.endswith("USDT") else f"{s}USDT"


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _date_tag(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def cache_path(
    symbol: str,
    kind: str,
    frequency: str,
    start: datetime,
    end: datetime,
    root: Optional[Path] = None,
) -> Path:
    """Return the canonical Parquet path for a ``(symbol, kind, freq)`` shard.

    ``kind`` is one of ``"spot"``, ``"funding"``, ``"oi"``. The filename
    layout deliberately encodes every dimension of the cache key so that
    a directory listing is self-documenting.
    """
    base = (root or _CACHE_ROOT)
    base.mkdir(parents=True, exist_ok=True)
    pair = _to_pair(symbol)
    return base / (
        f"{pair}_{kind}_{frequency}_{_date_tag(start)}_{_date_tag(end)}.parquet"
    )


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - corruption is rare
        logger.warning("cache read failed for %s: %s — refetching", path, exc)
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# Endpoint-specific batch parsers
# ---------------------------------------------------------------------------


_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _parse_klines(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=_KLINE_COLS).set_index(
            pd.DatetimeIndex([], tz="UTC", name="timestamp")
        )
    df = pd.DataFrame(raw, columns=_KLINE_COLS)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_base", "taker_buy_quote"]
    df[numeric] = df[numeric].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True).rename("timestamp")
    return df.drop(columns=["ignore"])


def _parse_funding(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(
            columns=["symbol", "funding_rate", "mark_price"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = pd.DataFrame(raw)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["mark_price"] = df.get("markPrice", pd.Series([np.nan] * len(df))).astype(float)
    df.index = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).rename(
        "timestamp"
    )
    return df[["symbol", "funding_rate", "mark_price"]]


def _parse_oi_hist(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(
            columns=["symbol", "open_interest", "open_interest_value"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = pd.DataFrame(raw)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df["open_interest_value"] = df["sumOpenInterestValue"].astype(float)
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True).rename("timestamp")
    return df[["symbol", "open_interest", "open_interest_value"]]


# ---------------------------------------------------------------------------
# Paginated fetchers
# ---------------------------------------------------------------------------


def _paginate(
    *,
    url: str,
    base_params: dict,
    parse: callable,
    start_ms: int,
    end_ms: int,
    page_size: int,
    advance_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Walk an endpoint from ``start_ms`` to ``end_ms`` in fixed pages.

    ``advance_ms`` lets fixed-grid endpoints (klines) advance by exactly
    ``page_size * interval``; variable-grid endpoints (funding, OI)
    advance by ``last_timestamp + 1`` to avoid duplicates.
    """
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = dict(base_params)
        params.update(startTime=cursor, endTime=end_ms, limit=page_size)
        raw = _http_get(url, params)
        df = parse(raw)
        if df.empty:
            break
        frames.append(df)
        if advance_ms is not None:
            cursor += page_size * advance_ms
        else:
            last_ms = int(df.index[-1].value // 1_000_000)
            if last_ms <= cursor:
                break
            cursor = last_ms + 1
        if len(df) < page_size:
            break
    if not frames:
        return parse([])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.loc[(out.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")) &
                   (out.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC"))]


def fetch_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    *,
    market: str = "spot",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV klines for ``symbol`` between ``start`` and ``end``."""
    if interval not in _INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")
    pair = _to_pair(symbol)
    kind = "spot" if market == "spot" else "futures"
    path = cache_path(pair, kind, interval, start, end)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    if market == "spot":
        url = f"{SPOT_BASE}/api/v3/klines"
    elif market == "futures":
        url = f"{FAPI_BASE}/fapi/v1/klines"
    else:
        raise ValueError(f"unsupported market: {market!r}")
    df = _paginate(
        url=url,
        base_params={"symbol": pair, "interval": interval},
        parse=_parse_klines,
        start_ms=_to_ms(start),
        end_ms=_to_ms(end),
        page_size=MAX_KLINES_PER_REQ,
        advance_ms=_INTERVAL_MS[interval],
    )
    _write_cache(path, df)
    return df


def fetch_funding_rate(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical USDT-perp funding rates (8h cadence)."""
    pair = _to_pair(symbol)
    path = cache_path(pair, "funding", "8h", start, end)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    df = _paginate(
        url=f"{FAPI_BASE}/fapi/v1/fundingRate",
        base_params={"symbol": pair},
        parse=_parse_funding,
        start_ms=_to_ms(start),
        end_ms=_to_ms(end),
        page_size=MAX_FUNDING_PER_REQ,
    )
    _write_cache(path, df)
    return df


def fetch_open_interest_hist(
    symbol: str,
    period: str,
    start: datetime,
    end: datetime,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical futures open interest.

    Binance only retains roughly the last 30 days of OI history, so
    calls for older windows (e.g. the Terra-Luna or FTX controls) will
    return an empty frame. That limitation is documented in
    ``data/README.md``.
    """
    pair = _to_pair(symbol)
    path = cache_path(pair, "oi", period, start, end)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    df = _paginate(
        url=f"{FAPI_BASE}/futures/data/openInterestHist",
        base_params={"symbol": pair, "period": period},
        parse=_parse_oi_hist,
        start_ms=_to_ms(start),
        end_ms=_to_ms(end),
        page_size=MAX_OI_HIST_PER_REQ,
    )
    _write_cache(path, df)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def fetch_all(
    windows: Iterable[Window] = tuple(WINDOWS.values()),
    symbols: Iterable[str] = SYMBOLS,
    *,
    include_funding: bool = True,
    include_oi: bool = True,
) -> None:
    """Fetch every ``(symbol, window)`` combination into the cache.

    Funding and OI are scoped per-window using the window's
    ``[start, end]``. OI history is silently empty for windows older
    than ~30 days from "now" (Binance retention).
    """
    for window in windows:
        for symbol in symbols:
            logger.info("spot klines: %s %s %s", symbol, window.interval, window.name)
            fetch_klines(symbol, window.interval, window.start, window.end)
            if include_funding:
                logger.info("funding: %s %s", symbol, window.name)
                fetch_funding_rate(symbol, window.start, window.end)
            if include_oi:
                # Map kline interval to OI period (OI supports 5m+).
                oi_period = window.interval if window.interval != "1m" else "5m"
                logger.info("OI hist: %s %s %s", symbol, oi_period, window.name)
                fetch_open_interest_hist(symbol, oi_period, window.start, window.end)


# ---------------------------------------------------------------------------
# Alignment + log returns
# ---------------------------------------------------------------------------


def _expected_grid(start: datetime, end: datetime, interval: str) -> pd.DatetimeIndex:
    freq = {"1m": "min", "5m": "5min", "15m": "15min", "1h": "h", "1d": "D"}.get(
        interval
    )
    if freq is None:
        raise ValueError(f"interval {interval!r} not supported for alignment")
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return pd.date_range(start=start, end=end, freq=freq, tz="UTC", inclusive="left")


_GAP_FILL_THRESHOLD = pd.Timedelta(minutes=5)


def load_aligned_returns(
    start: datetime,
    end: datetime,
    interval: str,
    *,
    symbols: Iterable[str] = SYMBOLS,
    market: str = "spot",
    fill_threshold: pd.Timedelta = _GAP_FILL_THRESHOLD,
) -> pd.DataFrame:
    """Load cached closes, align across symbols, compute log-returns.

    The full ``[start, end)`` grid for ``interval`` is constructed and
    every symbol is reindexed onto it. Missing bars whose run-length is
    strictly shorter than ``fill_threshold`` are forward-filled;
    anything longer raises a ``UserWarning`` with the gap location and
    duration so the caller can decide whether to drop the window.
    """
    grid = _expected_grid(start, end, interval)
    closes: dict[str, pd.Series] = {}
    for symbol in symbols:
        df = fetch_klines(symbol, interval, start, end, market=market)
        if df.empty:
            raise RuntimeError(
                f"no cached klines for {symbol} {interval} {start.date()}..{end.date()}"
            )
        close = df["close"].reindex(grid)
        _warn_on_large_gaps(symbol, close, fill_threshold)
        close = _fill_small_gaps(close, fill_threshold)
        closes[symbol.upper()] = close

    closes_df = pd.DataFrame(closes).dropna(how="any")
    log_returns = np.log(closes_df).diff().dropna(how="any")
    log_returns.index.name = "timestamp"
    return log_returns


def _consecutive_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, length)`` pairs for each run of ``True`` in ``mask``."""
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def _fill_small_gaps(
    series: pd.Series, threshold: pd.Timedelta
) -> pd.Series:
    """Forward-fill runs of NaN strictly shorter than ``threshold``."""
    if series.empty or not series.isna().any():
        return series
    idx = series.index
    if len(idx) < 2:
        return series
    step = idx[1] - idx[0]
    max_fill_steps = max(int(threshold / step) - 1, 0)
    if max_fill_steps <= 0:
        return series
    return series.ffill(limit=max_fill_steps)


def _warn_on_large_gaps(
    symbol: str, series: pd.Series, threshold: pd.Timedelta
) -> None:
    if series.empty or not series.isna().any():
        return
    idx = series.index
    if len(idx) < 2:
        return
    step = idx[1] - idx[0]
    runs = _consecutive_runs(series.isna().to_numpy())
    for start, length in runs:
        if length * step >= threshold:
            warnings.warn(
                f"{symbol}: gap of {length} bars ({length * step}) starting at "
                f"{idx[start]} exceeds {threshold} fill threshold",
                UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Public archive (data.binance.vision) — geo-unrestricted bulk klines
# ---------------------------------------------------------------------------
#
# The Binance REST API (`api.binance.com`) is geo-blocked from this
# execution environment (HTTP 451), but the public bulk archive at
# ``https://data.binance.vision/`` is a static S3-backed file server with
# no geo restriction. It serves the same kline data, packaged as monthly
# (and daily) ZIPped CSVs, e.g.
#
#     /data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-10.zip
#
# Files dated 2025 use *microsecond* open/close timestamps in column 0
# and column 6, not milliseconds (Binance changed the format mid-2025).
# `_parse_archive_csv` normalises this so callers always see UTC
# timestamps regardless of the underlying scale.

ARCHIVE_BASE = "https://data.binance.vision"

# Same column layout as the REST klines plus we keep ``ignore`` for parity.
_ARCHIVE_COLS = list(_KLINE_COLS)


def _archive_monthly_url(pair: str, interval: str, year: int, month: int) -> str:
    return (
        f"{ARCHIVE_BASE}/data/spot/monthly/klines/{pair}/{interval}/"
        f"{pair}-{interval}-{year:04d}-{month:02d}.zip"
    )


def _archive_cache_path(pair: str, interval: str, year: int, month: int) -> Path:
    base = _CACHE_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{pair}_archive_{interval}_{year:04d}{month:02d}.parquet"


def _detect_ts_scale(value: int) -> int:
    """Return the divisor that converts ``value`` to seconds.

    The Binance archive moved from millisecond to microsecond open/close
    timestamps in 2025. A clean heuristic by magnitude: anything north of
    ``1e15`` is microseconds, north of ``1e12`` is milliseconds, otherwise
    seconds. Inside the function so that ``_parse_archive_csv`` can pick
    the right ``pd.to_datetime`` unit per file.
    """
    if value > 1_000_000_000_000_000:
        return 1_000_000  # μs
    if value > 1_000_000_000_000:
        return 1_000  # ms
    return 1  # s


def _parse_archive_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse a single monthly archive CSV into the same shape as ``_parse_klines``."""
    from io import BytesIO

    df = pd.read_csv(BytesIO(csv_bytes), header=None, names=_ARCHIVE_COLS)
    if df.empty:
        return pd.DataFrame(columns=_ARCHIVE_COLS).set_index(
            pd.DatetimeIndex([], tz="UTC", name="timestamp")
        )
    # Strip an optional CSV header row if the archive includes one
    # (only some monthly bundles do).
    if not str(df["open_time"].iloc[0]).lstrip("-").isdigit():
        df = df.iloc[1:].reset_index(drop=True)

    open_time = df["open_time"].astype("int64")
    scale = _detect_ts_scale(int(open_time.iloc[0]))
    unit = {1_000_000: "us", 1_000: "ms", 1: "s"}[scale]
    df.index = pd.to_datetime(open_time, unit=unit, utc=True).rename("timestamp")

    numeric = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base", "taker_buy_quote",
    ]
    df[numeric] = df[numeric].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df = df.drop(columns=["ignore"])
    return df.sort_index()


def _download_archive_zip(url: str, *, timeout: float = 60.0) -> bytes:
    """Download a single archive ZIP, returning the inner CSV as bytes."""
    import zipfile
    from io import BytesIO

    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"archive HTTP {resp.status_code} for {url}")
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        # The archive contains a single .csv named after the symbol+interval+month.
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"unexpected archive contents: {names}")
        return zf.read(names[0])


def fetch_monthly_klines_archive(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch one month of klines for ``symbol`` from ``data.binance.vision``.

    Mirrors ``fetch_klines`` shape-wise (the same columns and a UTC
    DatetimeIndex named ``"timestamp"``) but pulls a single static ZIP
    rather than paginating REST. Cached on disk under ``data/raw/`` so
    repeat calls do no network I/O.
    """
    pair = _to_pair(symbol)
    path = _archive_cache_path(pair, interval, year, month)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    url = _archive_monthly_url(pair, interval, year, month)
    logger.info("archive: %s", url)
    csv_bytes = _download_archive_zip(url)
    df = _parse_archive_csv(csv_bytes)
    _write_cache(path, df)
    return df


def _iter_months(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Inclusive list of ``(year, month)`` covering ``[start, end]``."""
    y, m = start.year, start.month
    out: list[tuple[int, int]] = []
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def load_minute_returns_archive(
    start: datetime,
    end: datetime,
    *,
    symbols: Iterable[str] = SYMBOLS,
    fill_threshold: pd.Timedelta = _GAP_FILL_THRESHOLD,
) -> pd.DataFrame:
    """Load 1-minute closes from the public Binance archive and return log-returns.

    Aligns every symbol onto a 1-minute UTC grid, forward-fills NaN runs
    strictly shorter than ``fill_threshold`` (default 5 min), warns on
    anything longer, and returns ``log(close).diff()`` shaped
    ``(n_minutes, len(symbols))``.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    grid = _expected_grid(start, end, "1m")

    closes: dict[str, pd.Series] = {}
    for symbol in symbols:
        frames: list[pd.DataFrame] = []
        for y, m in _iter_months(start, end):
            frames.append(fetch_monthly_klines_archive(symbol, "1m", y, m))
        if not frames:
            raise RuntimeError(f"no archive data for {symbol}")
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df.loc[(df.index >= grid[0]) & (df.index <= grid[-1])]
        close = df["close"].reindex(grid)
        _warn_on_large_gaps(symbol, close, fill_threshold)
        close = _fill_small_gaps(close, fill_threshold)
        closes[symbol.upper()] = close

    closes_df = pd.DataFrame(closes).dropna(how="any")
    log_returns = np.log(closes_df).diff().dropna(how="any")
    log_returns.index.name = "timestamp"
    return log_returns


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover - exercised by hand
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fetch_all()


if __name__ == "__main__":  # pragma: no cover
    _cli()
