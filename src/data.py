"""
data.py
-------
Binance public API fetcher for:
  - Perpetual futures funding rate history (fapi.binance.com)
  - Perpetual futures OHLCV klines         (fapi.binance.com)
  - Spot OHLCV klines                      (api.binance.com)

No API key required for historical public data.
Rate-limiting: 1 200 weight/min on FAPI, 1 200 weight/min on API.
We stay well under by sleeping 0.25 s between requests and batching
the maximum 1 000 rows per call where the endpoint allows it.
"""

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAPI_BASE = "https://fapi.binance.com"
API_BASE  = "https://api.binance.com"

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "AVAXUSDT", "LINKUSDT", "ADAUSDT", "DOGEUSDT",
]

# Funding rates are settled every 8 hours: 00:00, 08:00, 16:00 UTC
FUNDING_INTERVAL_H = 8

# How many years of history to fetch
HISTORY_YEARS = 2

# Binance max rows per request
FUNDING_LIMIT = 1000
KLINE_LIMIT   = 1000

# Polite sleep between HTTP calls (seconds)
REQUEST_SLEEP = 0.25

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_dt(ms: int) -> datetime:
    """Convert Binance millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _dt_to_ms(dt: datetime) -> int:
    """Convert datetime (assumed UTC) to milliseconds epoch."""
    return int(dt.timestamp() * 1000)


def _get(url: str, params: dict, retries: int = 3) -> dict | list:
    """
    HTTP GET with simple retry / back-off logic.
    Raises RuntimeError after exhausting retries.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)   # exponential back-off: 2s, 4s
            else:
                raise RuntimeError(f"Failed to fetch {url} after {retries} attempts") from exc


# ---------------------------------------------------------------------------
# Funding Rate Fetcher
# ---------------------------------------------------------------------------

def fetch_funding_rates(
    symbol: str,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch the complete funding rate history for one perp symbol.

    Parameters
    ----------
    symbol    : e.g. "BTCUSDT"
    start_dt  : inclusive start (UTC).  Defaults to NOW - 2 years.
    end_dt    : inclusive end   (UTC).  Defaults to NOW.

    Returns
    -------
    DataFrame with columns:
        fundingTime (UTC datetime index), fundingRate (float), symbol (str)
    """
    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    if start_dt is None:
        start_dt = end_dt.replace(year=end_dt.year - HISTORY_YEARS)

    url    = f"{FAPI_BASE}/fapi/v1/fundingRate"
    cursor = _dt_to_ms(start_dt)
    end_ms = _dt_to_ms(end_dt)
    rows   = []

    logger.info("Fetching funding rates for %s from %s to %s", symbol, start_dt.date(), end_dt.date())

    while cursor < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": cursor,
            "endTime":   end_ms,
            "limit":     FUNDING_LIMIT,
        }
        data = _get(url, params)
        if not data:
            break

        rows.extend(data)
        # Advance cursor past the last timestamp we received
        last_ts = int(data[-1]["fundingTime"])
        cursor  = last_ts + 1   # +1 ms to avoid re-fetching the same row

        if len(data) < FUNDING_LIMIT:
            break   # no more pages

        time.sleep(REQUEST_SLEEP)

    if not rows:
        logger.warning("No funding data returned for %s", symbol)
        return pd.DataFrame(columns=["symbol", "fundingRate"])

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["symbol"]      = symbol
    df = df.set_index("fundingTime").sort_index()
    df = df[["symbol", "fundingRate"]]

    logger.info("  -> %d funding observations for %s", len(df), symbol)
    return df


def fetch_all_funding_rates(
    assets: list[str] = ASSETS,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch funding rates for all assets and return a combined wide DataFrame.

    Returns
    -------
    DataFrame indexed by fundingTime (UTC), columns = asset symbols, values = fundingRate.
    """
    frames = {}
    for symbol in assets:
        df = fetch_funding_rates(symbol, start_dt, end_dt)
        if not df.empty:
            frames[symbol] = df["fundingRate"]
        time.sleep(REQUEST_SLEEP)

    if not frames:
        raise RuntimeError("Could not fetch any funding rate data.")

    combined = pd.DataFrame(frames)
    combined.index.name = "fundingTime"
    return combined


# ---------------------------------------------------------------------------
# OHLCV Kline Fetcher
# ---------------------------------------------------------------------------

def _fetch_klines_single(
    url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list:
    """Paginate through kline endpoint and return raw list of rows."""
    rows   = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": cursor,
            "endTime":   end_ms,
            "limit":     KLINE_LIMIT,
        }
        data = _get(url, params)
        if not data:
            break

        rows.extend(data)
        last_open_ms = int(data[-1][0])
        cursor = last_open_ms + 1

        if len(data) < KLINE_LIMIT:
            break

        time.sleep(REQUEST_SLEEP)

    return rows


def _klines_to_df(rows: list, symbol: str) -> pd.DataFrame:
    """Convert raw kline list-of-lists to a clean OHLCV DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df["symbol"] = symbol
    return df


def fetch_perp_ohlcv(
    symbol: str,
    interval: str = "8h",
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a perpetual futures symbol from Binance FAPI.

    interval : Binance kline interval string, default "8h" to align with funding.
    """
    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    if start_dt is None:
        start_dt = end_dt.replace(year=end_dt.year - HISTORY_YEARS)

    url  = f"{FAPI_BASE}/fapi/v1/klines"
    rows = _fetch_klines_single(url, symbol, interval, _dt_to_ms(start_dt), _dt_to_ms(end_dt))
    df   = _klines_to_df(rows, symbol)
    logger.info("Fetched %d perp OHLCV rows for %s (%s)", len(df), symbol, interval)
    return df


def fetch_spot_ohlcv(
    symbol: str,
    interval: str = "8h",
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a spot symbol from Binance REST API.
    """
    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    if start_dt is None:
        start_dt = end_dt.replace(year=end_dt.year - HISTORY_YEARS)

    url  = f"{API_BASE}/api/v3/klines"
    rows = _fetch_klines_single(url, symbol, interval, _dt_to_ms(start_dt), _dt_to_ms(end_dt))
    df   = _klines_to_df(rows, symbol)
    logger.info("Fetched %d spot OHLCV rows for %s (%s)", len(df), symbol, interval)
    return df


# ---------------------------------------------------------------------------
# Convenience: load all data for the full universe
# ---------------------------------------------------------------------------

def load_universe(
    assets: list[str] = ASSETS,
    interval: str = "8h",
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> dict:
    """
    Fetch funding rates + perp OHLCV + spot OHLCV for every asset.

    Returns
    -------
    dict with keys:
        "funding"   -> wide DataFrame (fundingTime index, columns = symbols)
        "perp"      -> dict[symbol -> OHLCV DataFrame]
        "spot"      -> dict[symbol -> OHLCV DataFrame]
    """
    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    if start_dt is None:
        start_dt = end_dt.replace(year=end_dt.year - HISTORY_YEARS)

    print("Fetching funding rates for all assets...")
    funding = fetch_all_funding_rates(assets, start_dt, end_dt)

    perp_data = {}
    spot_data = {}

    for symbol in assets:
        print(f"  Fetching perp OHLCV: {symbol}")
        perp_data[symbol] = fetch_perp_ohlcv(symbol, interval, start_dt, end_dt)
        time.sleep(REQUEST_SLEEP)

        print(f"  Fetching spot OHLCV: {symbol}")
        spot_data[symbol] = fetch_spot_ohlcv(symbol, interval, start_dt, end_dt)
        time.sleep(REQUEST_SLEEP)

    return {"funding": funding, "perp": perp_data, "spot": spot_data}
