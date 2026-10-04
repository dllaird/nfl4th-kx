"""Read-only Kalshi Trade API v2 client (public market-data endpoints only).

Strictly read-only by design: this class only issues GET requests and has no order,
portfolio or authenticated methods. Do not add them.

Prices are returned as floats in dollars (0-1) and sizes as float contracts. Live and
`/historical/*` responses use different field names (`yes_price_dollars` vs
`yes_price`, `count_fp` vs `count`); `normalize_*` helpers map both to one schema.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Iterator
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


class RateLimiter:
    """Simple thread-safe spacing limiter (requests per second)."""

    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class KalshiClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        rate_per_sec: float = 8.0,
        max_retries: int = 8,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.limiter = RateLimiter(rate_per_sec)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # -- transport ---------------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self.base_url}{path}"
        params = {k: v for k, v in (params or {}).items() if v is not None}
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                err: Any = exc
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in (429, 500, 502, 503, 504):
                    resp.raise_for_status()
                err = f"HTTP {resp.status_code}"
            # Kalshi sends no Retry-After header: exponential backoff with jitter
            delay = min(60.0, 0.5 * 2**attempt) * (0.5 + random.random())
            log.warning("GET %s failed (%s); retry %d in %.1fs", path, err, attempt + 1, delay)
            time.sleep(delay)
        raise RuntimeError(f"GET {path} failed after {self.max_retries} retries")

    def paginate(self, path: str, key: str, params: dict[str, Any] | None = None) -> Iterator[dict]:
        params = dict(params or {})
        while True:
            body = self.get(path, params)
            yield from body.get(key) or []
            cursor = body.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # -- discovery ---------------------------------------------------------------

    def historical_cutoff(self) -> dict:
        return self.get("/historical/cutoff")

    def get_series(self, series_ticker: str) -> dict:
        return self.get(f"/series/{series_ticker}")["series"]

    def iter_events(self, series_ticker: str, status: str | None = None) -> Iterator[dict]:
        return self.paginate(
            "/events", "events", {"series_ticker": series_ticker, "status": status, "limit": 200}
        )

    def iter_markets(
        self, *, series_ticker: str | None = None, event_ticker: str | None = None, historical: bool = False
    ) -> Iterator[dict]:
        path = "/historical/markets" if historical else "/markets"
        params = {"series_ticker": series_ticker, "event_ticker": event_ticker, "limit": 1000}
        return (normalize_market(m) for m in self.paginate(path, "markets", params))

    # -- market data -------------------------------------------------------------

    def get_orderbook(self, ticker: str, depth: int = 0) -> dict:
        return self.get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_orderbooks(self, tickers: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(tickers), 100):
            body = self.get("/markets/orderbooks", {"tickers": tickers[i : i + 100]})
            out.extend(body.get("orderbooks") or [])
        return out

    def iter_trades(
        self, ticker: str, min_ts: int | None = None, max_ts: int | None = None, historical: bool = False
    ) -> Iterator[dict]:
        """Trades for one market, newest first. Timestamps are Unix seconds."""
        path = "/historical/trades" if historical else "/markets/trades"
        params = {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts, "limit": 1000}
        return (normalize_trade(t) for t in self.paginate(path, "trades", params))

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        historical: bool = False,
    ) -> list[dict]:
        """Candles (period 1, 60 or 1440 minutes); chunks requests to <= 5000 candles."""
        if period_interval not in (1, 60, 1440):
            raise ValueError("period_interval must be 1, 60 or 1440")
        path = (
            f"/historical/markets/{ticker}/candlesticks"
            if historical
            else f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        )
        step = 5000 * 60 * period_interval
        out: list[dict] = []
        for lo in range(start_ts, end_ts, step):
            hi = min(end_ts, lo + step)
            body = self.get(path, {"start_ts": lo, "end_ts": hi, "period_interval": period_interval})
            out.extend(normalize_candle(c) for c in body.get("candlesticks") or [])
        return out


# -- response normalization -------------------------------------------------------


def _f(d: dict, *keys: str) -> float | None:
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return float(v)
    return None


def normalize_trade(t: dict) -> dict:
    return {
        "trade_id": t.get("trade_id"),
        "ticker": t.get("ticker"),
        "created_time": t.get("created_time"),
        "yes_price": _f(t, "yes_price_dollars", "yes_price"),
        "count": _f(t, "count_fp", "count"),
        "taker_side": t.get("taker_side"),
        "is_block_trade": t.get("is_block_trade"),
    }


def _ohlc(c: dict, key: str) -> dict:
    block = c.get(key) or {}
    return {
        f"{key}_{f}": _f(block, f"{f}_dollars", f)
        for f in ("open", "high", "low", "close")
    }


def normalize_candle(c: dict) -> dict:
    out = {
        "end_period_ts": c.get("end_period_ts"),
        "volume": _f(c, "volume_fp", "volume"),
        "open_interest": _f(c, "open_interest_fp", "open_interest"),
    }
    for key in ("price", "yes_bid", "yes_ask"):
        out.update(_ohlc(c, key))
    price = c.get("price") or {}
    out["price_mean"] = _f(price, "mean_dollars", "mean")
    return out


_MARKET_FIELDS = (
    "ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title", "strike_type",
    "status", "result", "open_time", "close_time", "settlement_ts", "custom_strike",
)


def normalize_market(m: dict) -> dict:
    out = {k: m.get(k) for k in _MARKET_FIELDS}
    out["floor_strike"] = _f(m, "floor_strike")
    out["cap_strike"] = _f(m, "cap_strike")
    out["volume"] = _f(m, "volume_fp", "volume")
    out["open_interest"] = _f(m, "open_interest_fp", "open_interest")
    out["settlement_value"] = _f(m, "settlement_value_dollars", "settlement_value")
    out["last_price"] = _f(m, "last_price_dollars", "last_price")
    team = (m.get("custom_strike") or {}).get("football_team")
    out["football_team"] = team
    return out
