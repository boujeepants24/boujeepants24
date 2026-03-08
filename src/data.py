"""
data.py
-------
Handles all data acquisition and preprocessing.

Fetches OHLCV data via yfinance for the equity + crypto universe,
aligns on a common daily calendar, computes log returns, and
exposes clean DataFrames for downstream factor computation.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Universe definition
# ---------------------------------------------------------------------------

EQUITIES = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "JPM",  "GS",   "XOM",  "CVX",  "JNJ",  "UNH",
    "TSLA", "AMD",  "NFLX", "COST", "WMT",  "HD",
    "V",    "MA",
]

CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]

ALL_TICKERS = EQUITIES + CRYPTO
BENCHMARK   = "SPY"

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download a single ticker with yfinance and return a clean OHLCV frame.
    Returns an empty DataFrame on failure.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            print(f"  [WARN] No data returned for {ticker}")
            return pd.DataFrame()

        # yfinance may return MultiIndex columns when downloading one ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as exc:
        print(f"  [WARN] Failed to download {ticker}: {exc}")
        return pd.DataFrame()


def fetch_all(start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for the full universe + SPY.
    Returns a dict keyed by ticker symbol.
    """
    print("Fetching price data from yfinance ...")
    raw: dict[str, pd.DataFrame] = {}

    for ticker in ALL_TICKERS + [BENCHMARK]:
        print(f"  -> {ticker}")
        df = _download_ticker(ticker, start, end)
        if not df.empty:
            raw[ticker] = df

    print(f"  Downloaded {len(raw)} tickers successfully.")
    return raw


# ---------------------------------------------------------------------------
# Alignment & return computation
# ---------------------------------------------------------------------------

def build_close_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract adjusted close prices and align on the intersection of all
    trading dates (handles the equity/crypto calendar mismatch by forward-
    filling crypto on equity holiday gaps, then inner-joining on equity dates).

    Strategy:
      1. Build a combined close panel (union of all dates).
      2. Forward-fill up to 3 days to bridge weekend/holiday gaps.
      3. Reindex to the equity date index (drops crypto-only weekend rows).
    """
    close_frames = {}
    for ticker, df in raw.items():
        if "Close" in df.columns:
            close_frames[ticker] = df["Close"].rename(ticker)

    # Union of all dates
    panel = pd.concat(close_frames.values(), axis=1)
    panel.columns = list(close_frames.keys())
    panel.index = pd.to_datetime(panel.index)
    panel.sort_index(inplace=True)

    # Forward-fill gaps (crypto weekends, market holidays) up to 3 sessions
    panel = panel.ffill(limit=3)

    # Restrict to dates where SPY traded (equity business-day calendar)
    if BENCHMARK in panel.columns:
        equity_dates = panel[BENCHMARK].dropna().index
        panel = panel.reindex(equity_dates)

    # Drop tickers with >20% missing data after alignment
    missing_frac = panel.isna().mean()
    bad_tickers = missing_frac[missing_frac > 0.20].index.tolist()
    if bad_tickers:
        print(f"  [WARN] Dropping tickers with >20pct missing data: {bad_tickers}")
        panel.drop(columns=bad_tickers, inplace=True)

    # Fill residual gaps with forward-fill then backward-fill
    panel = panel.ffill().bfill()

    print(
        f"  Price panel shape: {panel.shape[0]} dates x {panel.shape[1]} assets, "
        f"{panel.index[0].date()} to {panel.index[-1].date()}"
    )
    return panel


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from close prices."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.iloc[1:]  # drop the first NaN row


def build_volume_panel(raw: dict[str, pd.DataFrame], date_index: pd.Index) -> pd.DataFrame:
    """
    Build a volume panel aligned to the equity date index.
    Used as a rough liquidity proxy inside the Quality factor.
    """
    vol_frames = {}
    for ticker, df in raw.items():
        if "Volume" in df.columns:
            vol_frames[ticker] = df["Volume"].rename(ticker)

    panel = pd.concat(vol_frames.values(), axis=1)
    panel.columns = list(vol_frames.keys())
    panel.index = pd.to_datetime(panel.index)
    panel.sort_index(inplace=True)
    panel = panel.ffill(limit=3)
    panel = panel.reindex(date_index)
    panel = panel.ffill().bfill()
    return panel


# ---------------------------------------------------------------------------
# Month-end index helper
# ---------------------------------------------------------------------------

def get_month_ends(date_index: pd.Index) -> pd.Index:
    """Return the last trading day of each calendar month within date_index."""
    return date_index[pd.Series(date_index).dt.is_month_end.values |
                      (pd.Series(date_index).dt.month.values !=
                       pd.Series(date_index).shift(-1).dt.month.values)]


# ---------------------------------------------------------------------------
# Main data-loading entry point
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Single object that holds all preprocessed data used by the backtest.

    Attributes
    ----------
    close   : pd.DataFrame  -- daily close prices, shape (T, N)
    returns : pd.DataFrame  -- daily log returns,  shape (T-1, N)
    volume  : pd.DataFrame  -- daily volume,        shape (T, N)
    bench   : pd.Series     -- SPY log returns
    month_ends : pd.Index   -- last trading day of each month
    tickers : list[str]     -- asset universe (excl. SPY)
    """

    def __init__(self, start: str = "2018-01-01", end: str = "2024-12-31"):
        self.start = start
        self.end   = end

        raw = fetch_all(start, end)

        self.close = build_close_panel(raw)

        # Separate benchmark from universe
        self.bench_close = self.close.pop(BENCHMARK) if BENCHMARK in self.close.columns else None
        self.tickers = [t for t in self.close.columns if t in ALL_TICKERS]
        self.close = self.close[self.tickers]

        self.returns   = compute_returns(self.close)
        self.volume    = build_volume_panel(raw, self.close.index)
        self.volume    = self.volume[self.tickers]

        if self.bench_close is not None:
            bench_log = np.log(self.bench_close / self.bench_close.shift(1)).iloc[1:]
            self.bench = bench_log.reindex(self.returns.index).fillna(0.0)
        else:
            self.bench = pd.Series(0.0, index=self.returns.index)

        self.month_ends = get_month_ends(self.close.index)

        print(f"  Universe: {len(self.tickers)} assets | "
              f"{self.returns.shape[0]} daily return observations")
