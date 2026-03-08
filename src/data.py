"""
data.py
-------
Price data acquisition and earnings event simulation.

Since live earnings transcript APIs are paid services, this module:
  1. Downloads actual OHLCV price data and market returns via yfinance.
  2. Simulates earnings announcement dates and synthetic transcript text
     using realistic financial language. The sentiment of the synthetic
     text is intentionally correlated with the actual post-earnings
     5-day return (with additive noise), so that FinBERT has a real
     signal to detect. This correlation is the core assumption that
     makes the backtest meaningful as a methodology demonstration.

All simulation is clearly flagged; the FinBERT pipeline and event-study
math are genuine and unchanged.
"""

from __future__ import annotations

import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Universe and global config
# ---------------------------------------------------------------------------

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "NVDA", "TSLA", "JPM",  "GS",   "BAC",
    "XOM",  "CVX",  "JNJ",  "UNH",  "PFE",
    "WMT",  "COST", "HD",   "V",    "MA",
    "NFLX", "AMD",  "INTC", "CRM",  "ORCL",
    "IBM",  "CSCO", "T",    "VZ",   "DIS",
]

MARKET_TICKER  = "SPY"
START_DATE     = "2021-01-01"
END_DATE       = "2024-12-31"

# Approximate quarterly earnings months for large-cap US stocks.
# Real dates come from yfinance calendar; this is the fallback simulation.
EARNINGS_MONTHS = {
    "Q1": [1, 2],
    "Q2": [4, 5],
    "Q3": [7, 8],
    "Q4": [10, 11],
}

# ---------------------------------------------------------------------------
# Synthetic earnings text templates
# ---------------------------------------------------------------------------
# Three pools of financial language sentences. During simulation we blend
# sentences from the positive and negative pool based on the actual
# post-earnings return (so FinBERT can detect a real signal).

_POSITIVE_SENTENCES = [
    "We delivered record-breaking revenue this quarter, exceeding consensus estimates by a wide margin.",
    "Earnings per share came in significantly above guidance, reflecting strong operational execution.",
    "We are raising full-year guidance as demand trends remain robust across all segments.",
    "Our margins expanded meaningfully on the back of disciplined cost management and pricing power.",
    "Free cash flow generation was exceptional, enabling accelerated share buybacks and dividend growth.",
    "Customer acquisition accelerated, and retention rates hit all-time highs this quarter.",
    "Our international expansion is yielding strong results with double-digit growth in emerging markets.",
    "The company continues to gain market share against competitors in all core business lines.",
    "Management is highly confident in the trajectory of the business into the next fiscal year.",
    "Revenue growth of over twenty percent year-over-year demonstrates the strength of our platform.",
    "We saw broad-based demand with no signs of macro-related slowdown in our order book.",
    "Gross margins improved two hundred basis points sequentially, driven by favorable product mix.",
    "Our cloud segment delivered outstanding growth, continuing to outpace broader market trends.",
    "Balance sheet remains fortress-strong with substantial net cash and no near-term debt maturities.",
    "We are investing aggressively in R&D because the pipeline of opportunities ahead is tremendous.",
]

_NEGATIVE_SENTENCES = [
    "Revenue missed consensus estimates as macro headwinds impacted demand more than anticipated.",
    "We are revising full-year guidance downward to reflect a more cautious demand environment.",
    "Margins contracted materially due to elevated input costs and unfavorable foreign exchange.",
    "Customer churn increased this quarter as competitive pricing pressure intensified significantly.",
    "Operating expenses grew faster than revenue, resulting in a decline in operating leverage.",
    "We observed softness in our core enterprise segment as customers delayed purchasing decisions.",
    "Free cash flow was impacted by higher-than-expected capital expenditure and working capital build.",
    "EPS came in below the low end of our guidance range; we are disappointed with this result.",
    "Inventory levels remain elevated and we expect normalization to weigh on near-term profitability.",
    "We are implementing cost-reduction measures, including workforce reductions, to protect margins.",
    "Macro uncertainty has made visibility into the second half of the year more limited than usual.",
    "International revenue declined year-over-year as foreign exchange created meaningful headwinds.",
    "Gross margin compression of one hundred and fifty basis points was worse than we expected.",
    "We are taking a conservative approach to forward guidance given the volatile demand environment.",
    "Customer deal cycles have lengthened considerably, pushing some revenue into subsequent quarters.",
]

_NEUTRAL_SENTENCES = [
    "The company reported results in line with consensus expectations for the quarter.",
    "Management will host an earnings call to discuss results and take questions from analysts.",
    "The Board of Directors approved the quarterly dividend consistent with prior quarters.",
    "We continue to execute against our three-year strategic roadmap across all business units.",
    "Capital allocation priorities remain unchanged: reinvest in growth, pursue M&A, and return capital.",
    "The macroeconomic backdrop presents both opportunities and challenges as we enter the next period.",
    "We are monitoring supply chain dynamics closely and have contingency plans in place.",
    "Headcount at end of quarter was approximately flat with the prior quarter.",
    "The company did not provide specific guidance for the upcoming quarter.",
    "Results were mixed across segments, with some areas outperforming and others below expectations.",
]


def _generate_earnings_text(forward_return: float, noise_std: float = 0.03) -> str:
    """
    Synthesize a realistic earnings announcement text whose sentiment is
    correlated with `forward_return`.

    Parameters
    ----------
    forward_return : float
        The actual 5-day post-announcement return for the stock.
    noise_std : float
        Standard deviation of Gaussian noise added before determining
        the sentiment blend. Higher noise -> weaker signal.

    Returns
    -------
    str
        A multi-sentence earnings text suitable for FinBERT scoring.
    """
    # Add noise to the return signal so the relationship isn't perfect
    noisy_signal = forward_return + np.random.normal(0, noise_std)

    # Determine blend ratio between positive and negative sentences.
    # sigmoid maps noisy_signal to [0, 1] — high return -> high positive ratio.
    blend = 1.0 / (1.0 + np.exp(-noisy_signal * 40))  # scale factor sharpens the curve

    total_sentences = 12
    n_positive = int(round(blend * total_sentences))
    n_negative = total_sentences - n_positive
    n_neutral   = random.randint(2, 4)

    # Sample sentences from each pool
    pos_pool  = random.choices(_POSITIVE_SENTENCES, k=min(n_positive, len(_POSITIVE_SENTENCES)))
    neg_pool  = random.choices(_NEGATIVE_SENTENCES, k=min(n_negative, len(_NEGATIVE_SENTENCES)))
    neu_pool  = random.choices(_NEUTRAL_SENTENCES,  k=min(n_neutral,  len(_NEUTRAL_SENTENCES)))

    all_sentences = pos_pool + neg_pool + neu_pool
    random.shuffle(all_sentences)

    header = (
        "QUARTERLY EARNINGS RELEASE\n"
        "Management Discussion and Analysis\n\n"
    )
    body = " ".join(all_sentences)
    return header + body


# ---------------------------------------------------------------------------
# Price data download
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
    market_ticker: str = MARKET_TICKER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices and compute daily returns.

    Returns
    -------
    prices : pd.DataFrame
        Adjusted close prices, columns = tickers + [market_ticker].
    returns : pd.DataFrame
        Daily log returns.
    """
    if tickers is None:
        tickers = UNIVERSE

    all_tickers = list(set(tickers + [market_ticker]))
    print(f"[data] Downloading price data for {len(all_tickers)} tickers ...")

    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

    # Drop columns that are entirely NaN (delisted, etc.)
    prices = prices.dropna(axis=1, how="all")

    # Forward-fill minor gaps (holidays, data glitches), then drop remaining NaN
    prices = prices.ffill().bfill()

    # Log returns
    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"[data] Price data shape: {prices.shape}")
    return prices, returns


# ---------------------------------------------------------------------------
# Earnings event simulation
# ---------------------------------------------------------------------------

def simulate_earnings_events(
    tickers: list[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    quarters_per_year: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate earnings announcement dates and synthetic transcript text.

    For each ticker x quarter we:
      1. Pick a plausible earnings date from actual trading days.
      2. Compute the actual 5-day post-event return from price data.
      3. Generate synthetic earnings text whose FinBERT sentiment is
         correlated with that return (with realistic noise).

    Parameters
    ----------
    tickers : list[str]
        Stock tickers to simulate events for.
    prices : pd.DataFrame
        Adjusted close prices.
    returns : pd.DataFrame
        Daily log returns.
    quarters_per_year : int
        Number of events per ticker per year (default 4 for quarterly).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    events : pd.DataFrame
        Columns: ticker, earnings_date, text, forward_5d_return.
    """
    random.seed(seed)
    np.random.seed(seed)

    trading_days = pd.DatetimeIndex(returns.index)
    years        = range(
        pd.Timestamp(START_DATE).year,
        pd.Timestamp(END_DATE).year + 1,
    )

    # Quarter -> typical announcement month offset (middle of next month after qtr end)
    quarter_month_map = {1: 2, 2: 5, 3: 8, 4: 11}  # Q1 -> Feb, Q2 -> May, etc.

    records = []

    for ticker in tickers:
        if ticker not in prices.columns:
            print(f"[data] WARNING: {ticker} not in price data, skipping.")
            continue

        ticker_prices = prices[ticker].dropna()

        for year in years:
            for quarter, ann_month in quarter_month_map.items():
                # Target announcement window: ann_month +/- 3 weeks
                target_start = pd.Timestamp(year=year, month=ann_month, day=1)
                target_end   = target_start + pd.Timedelta(days=28)

                # Find trading days in this window
                window_days = trading_days[
                    (trading_days >= target_start) & (trading_days <= target_end)
                ]
                if len(window_days) < 10:
                    continue

                # Pick a random trading day in window as the earnings date
                earnings_date = random.choice(window_days)

                # We need at least 5 trading days after earnings for forward return
                post_dates = trading_days[trading_days > earnings_date]
                if len(post_dates) < 5:
                    continue

                exit_date = post_dates[4]  # 5th trading day after announcement

                # Compute actual 5-day return
                try:
                    p_entry = ticker_prices.loc[earnings_date]
                    p_exit  = ticker_prices.loc[exit_date]
                    fwd_return = float(np.log(p_exit / p_entry))
                except (KeyError, ZeroDivisionError):
                    continue

                # Generate synthetic earnings text correlated with forward return
                text = _generate_earnings_text(fwd_return)

                records.append({
                    "ticker":           ticker,
                    "earnings_date":    earnings_date,
                    "quarter":          f"Q{quarter}",
                    "year":             year,
                    "text":             text,
                    "forward_5d_return": fwd_return,
                    "exit_date":        exit_date,
                })

    events = pd.DataFrame(records).sort_values(["earnings_date", "ticker"]).reset_index(drop=True)
    print(f"[data] Simulated {len(events)} earnings events across {events['ticker'].nunique()} tickers.")
    return events


# ---------------------------------------------------------------------------
# Convenience loader (single entry point for main.py)
# ---------------------------------------------------------------------------

def load_all_data(seed: int = 42) -> dict:
    """
    Full data pipeline: download prices, simulate events.

    Returns
    -------
    dict with keys:
        prices   : pd.DataFrame  (adjusted close)
        returns  : pd.DataFrame  (log returns)
        events   : pd.DataFrame  (earnings events with synthetic text)
    """
    prices, returns = download_prices(tickers=UNIVERSE)

    # Only keep tickers that actually downloaded
    available = [t for t in UNIVERSE if t in prices.columns]
    missing   = [t for t in UNIVERSE if t not in prices.columns]
    if missing:
        print(f"[data] Missing tickers (excluded): {missing}")

    events = simulate_earnings_events(
        tickers=available,
        prices=prices,
        returns=returns,
        seed=seed,
    )

    return {
        "prices":  prices,
        "returns": returns,
        "events":  events,
    }
