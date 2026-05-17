"""Tests for ``tda_oct10.data_ingestion``.

These tests never hit the real Binance API. The single HTTP chokepoint
(``data_ingestion._http_get``) is monkey-patched to return canned kline
batches with deterministic, gap-free OHLCV. Cache behavior is verified
by counting calls to the mock.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tda_oct10 import data_ingestion as di  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _utc(year, month, day, hour=0):
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _make_klines_batch(start_ms: int, end_ms: int, interval_ms: int) -> list[list]:
    """Return Binance-shaped klines spanning ``[start_ms, end_ms]`` inclusive.

    Prices walk smoothly so log-returns are well-defined and finite.
    """
    rows = []
    t = start_ms
    base_price = 100.0
    i = 0
    while t <= end_ms:
        # Smooth deterministic walk; bounded so log-returns stay reasonable.
        close = base_price * (1.0 + 0.001 * np.sin(i / 10.0))
        open_ = close * (1.0 - 0.0001)
        high = close * 1.0005
        low = close * 0.9995
        rows.append([
            t,
            f"{open_:.8f}",
            f"{high:.8f}",
            f"{low:.8f}",
            f"{close:.8f}",
            "10.0",
            t + interval_ms - 1,
            "1000.0",
            5,
            "5.0",
            "500.0",
            "0",
        ])
        t += interval_ms
        i += 1
    return rows


class MockHTTP:
    """Stand-in for ``_http_get`` that records calls."""

    def __init__(self, interval_ms: int):
        self.interval_ms = interval_ms
        self.calls: list[dict] = []

    def __call__(self, url, params, **_):
        self.calls.append({"url": url, "params": dict(params)})
        if "klines" in url:
            start = int(params["startTime"])
            end = int(params["endTime"])
            limit = int(params.get("limit", 1000))
            # Cap the batch to ``limit`` rows so pagination logic exercises.
            batch_end = min(end, start + (limit - 1) * self.interval_ms)
            return _make_klines_batch(start, batch_end, self.interval_ms)
        if "fundingRate" in url:
            return []
        if "openInterestHist" in url:
            return []
        raise AssertionError(f"unexpected URL in test: {url}")


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache root at a temp dir for every test."""
    di.set_cache_root(tmp_path)
    yield tmp_path
    di.set_cache_root(di._DEFAULT_CACHE_ROOT)


@pytest.fixture
def mock_http_1h(monkeypatch):
    """Patch ``_http_get`` with a 1-hour-interval mock."""
    mock = MockHTTP(interval_ms=di._INTERVAL_MS["1h"])
    monkeypatch.setattr(di, "_http_get", mock)
    return mock


# ---------------------------------------------------------------------------
# fetch_klines: parsing + cache
# ---------------------------------------------------------------------------


def test_fetch_klines_returns_parsed_dataframe(isolated_cache, mock_http_1h):
    start, end = _utc(2025, 10, 1), _utc(2025, 10, 1, 5)
    df = di.fetch_klines("BTC", "1h", start, end)
    assert not df.empty
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing
    assert df["close"].dtype == float
    assert (df["close"] > 0).all()


def test_fetch_klines_second_call_does_not_hit_api(isolated_cache, mock_http_1h):
    start, end = _utc(2025, 10, 1), _utc(2025, 10, 1, 5)
    di.fetch_klines("BTC", "1h", start, end)
    calls_after_first = len(mock_http_1h.calls)
    assert calls_after_first > 0

    df2 = di.fetch_klines("BTC", "1h", start, end)
    assert len(mock_http_1h.calls) == calls_after_first, (
        "second call must be served from cache"
    )
    assert not df2.empty


def test_fetch_klines_use_cache_false_refetches(isolated_cache, mock_http_1h):
    start, end = _utc(2025, 10, 1), _utc(2025, 10, 1, 5)
    di.fetch_klines("BTC", "1h", start, end)
    before = len(mock_http_1h.calls)
    di.fetch_klines("BTC", "1h", start, end, use_cache=False)
    assert len(mock_http_1h.calls) > before


# ---------------------------------------------------------------------------
# load_aligned_returns: NaN-free, monotonic, unique
# ---------------------------------------------------------------------------


def test_load_aligned_returns_invariants(isolated_cache, mock_http_1h):
    start, end = _utc(2025, 10, 1), _utc(2025, 10, 1, 12)
    returns = di.load_aligned_returns(
        start, end, "1h", symbols=("BTC", "ETH", "SOL")
    )

    # No NaNs anywhere.
    assert not returns.isna().any().any(), "aligned returns must be NaN-free"

    # Monotonic + unique timestamps.
    assert returns.index.is_monotonic_increasing
    assert returns.index.is_unique

    # One column per requested symbol.
    assert set(returns.columns) == {"BTC", "ETH", "SOL"}

    # Log-returns should be finite and small for our synthetic prices.
    assert np.isfinite(returns.to_numpy()).all()
    assert (returns.abs() < 0.1).all().all()


def test_load_aligned_returns_uses_cache_on_second_call(
    isolated_cache, mock_http_1h
):
    start, end = _utc(2025, 10, 1), _utc(2025, 10, 1, 12)
    di.load_aligned_returns(start, end, "1h", symbols=("BTC", "ETH"))
    after_first = len(mock_http_1h.calls)

    di.load_aligned_returns(start, end, "1h", symbols=("BTC", "ETH"))
    assert len(mock_http_1h.calls) == after_first, (
        "load_aligned_returns must not refetch from the network on a warm cache"
    )


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------


def test_small_gap_is_forward_filled():
    idx = pd.date_range("2025-10-01", periods=10, freq="min", tz="UTC")
    s = pd.Series([1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
    filled = di._fill_small_gaps(s, pd.Timedelta(minutes=5))
    assert not filled.isna().any()
    # 2-minute gap was filled by carrying forward 2.0.
    assert filled.iloc[2] == 2.0
    assert filled.iloc[3] == 2.0


def test_large_gap_emits_warning():
    idx = pd.date_range("2025-10-01", periods=20, freq="min", tz="UTC")
    values = [1.0] * 20
    # 7-minute gap > 5-minute threshold.
    for k in range(5, 12):
        values[k] = np.nan
    s = pd.Series(values, index=idx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        di._warn_on_large_gaps("BTC", s, pd.Timedelta(minutes=5))
    assert any("gap" in str(w.message) for w in caught), (
        "large gaps must produce a UserWarning"
    )


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_rate_limiter_acquires_under_limit():
    limiter = di.RateLimiter(max_per_minute=10)
    # Ten quick acquires should not block; we only assert no exception
    # and that the internal window now has ten entries.
    for _ in range(10):
        limiter.acquire()
    assert len(limiter._window) == 10


def test_rate_limiter_rejects_invalid_limit():
    with pytest.raises(ValueError):
        di.RateLimiter(max_per_minute=0)


# ---------------------------------------------------------------------------
# cache_path encoding
# ---------------------------------------------------------------------------


def test_cache_path_encodes_all_dimensions(isolated_cache):
    p = di.cache_path("BTC", "spot", "1h", _utc(2025, 8, 1), _utc(2025, 11, 30))
    assert p.name == "BTCUSDT_spot_1h_20250801_20251130.parquet"
    assert p.parent == isolated_cache


def test_symbol_normalisation():
    assert di._to_pair("btc") == "BTCUSDT"
    assert di._to_pair("BTC") == "BTCUSDT"
    assert di._to_pair("BTCUSDT") == "BTCUSDT"
