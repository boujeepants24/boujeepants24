"""
data.py
-------
Fetches adjusted closing prices for the universe and SPY via yfinance.
Returns a cleaned DataFrame of daily returns, dropping tickers with insufficient history.
"""

import yfinance as yf
import pandas as pd
import numpy as np

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AMZN", "NFLX", "TSLA", "CRM",
    "JPM", "GS", "MS", "BAC", "WFC", "BLK", "V", "MA", "AXP", "USB",
    "XOM", "CVX", "COP", "SLB", "PSX", "VLO", "HAL", "MPC", "OXY", "PXD",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD", "CVS",
]

BENCHMARK = "SPY"
START_DATE = "2014-01-01"   # extra year for warm-up before 2015
END_DATE   = "2024-12-31"
MIN_HISTORY_DAYS = 252


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    # yfinance >= 0.2.x returns a MultiIndex (Price, Ticker) for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single ticker or flat column structure
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            prices.columns = [tickers[0]] if len(tickers) == 1 else tickers
        else:
            prices = raw.copy()

    prices = prices.sort_index()
    return prices


def load_data(
    universe: list[str] = UNIVERSE,
    benchmark: str = BENCHMARK,
    start: str = START_DATE,
    end: str = END_DATE,
    min_history: int = MIN_HISTORY_DAYS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Main entry point.

    Returns
    -------
    returns : pd.DataFrame
        Daily log returns for the surviving universe tickers.
    spy_returns : pd.Series
        Daily log returns for the benchmark (SPY).
    """
    print("Fetching price data from yfinance ...")
    all_tickers = universe + [benchmark]
    prices = fetch_prices(all_tickers, start, end)

    # Separate benchmark
    spy_prices = prices[benchmark].copy()
    stock_prices = prices.drop(columns=[benchmark])

    # Drop tickers with too few observations
    valid_cols = [
        col for col in stock_prices.columns
        if stock_prices[col].dropna().shape[0] >= min_history
    ]
    dropped = set(stock_prices.columns) - set(valid_cols)
    if dropped:
        print(f"  Dropped tickers (insufficient history): {sorted(dropped)}")
    stock_prices = stock_prices[valid_cols]

    # Forward-fill up to 5 trading days (handles isolated NaNs / halts)
    stock_prices = stock_prices.ffill(limit=5)

    # Compute log returns
    returns = np.log(stock_prices / stock_prices.shift(1)).dropna(how="all")
    spy_returns = np.log(spy_prices / spy_prices.shift(1)).dropna()

    # Align indices
    common_idx = returns.index.intersection(spy_returns.index)
    returns = returns.loc[common_idx]
    spy_returns = spy_returns.loc[common_idx]

    print(f"  Universe after filtering: {len(returns.columns)} tickers")
    print(f"  Date range : {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Observations: {len(returns)}")

    return returns, spy_returns
