"""
factors.py
----------
Computes four cross-sectional equity factors at each monthly rebalance date:

  1. Momentum      -- 12-1 month cumulative log return (skip last month)
  2. Quality       -- Sharpe-ratio proxy: trailing 63-day mean / std of returns
                     (a pure-price Quality signal; no accounting data needed)
  3. Low-Volatility -- negative of realized 63-day daily return volatility
  4. Value          -- negative of price normalized by 252-day trailing high
                       (cheap = far from 52-week high = potential mean-revert)

All raw factor scores are:
  a) winsorized at the 5th / 95th percentile (removes extreme outliers)
  b) cross-sectionally z-scored (mean=0, std=1 each month)

The combined score is a weighted sum of z-scores; weights are supplied
externally (by the regime module) so this file is regime-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Individual factor computation
# ---------------------------------------------------------------------------

def factor_momentum(close: pd.DataFrame, date: pd.Timestamp, lookback: int = 252,
                    skip: int = 21) -> pd.Series:
    """
    12-1 month momentum: cumulative return from (date - lookback) to
    (date - skip), excluding the most recent month to avoid short-term reversal.

    Parameters
    ----------
    close    : full close-price history
    date     : rebalance date
    lookback : ~252 trading days = ~12 months
    skip     : ~21 trading days  = ~1 month (skip period)
    """
    loc = close.index.get_loc(date)

    if loc < lookback:
        return pd.Series(np.nan, index=close.columns)

    price_start = close.iloc[loc - lookback]
    price_end   = close.iloc[loc - skip]

    mom = np.log(price_end / price_start)
    return mom


def factor_quality(returns: pd.DataFrame, date: pd.Timestamp,
                   window: int = 63) -> pd.Series:
    """
    Quality proxy: trailing Sharpe-ratio of daily returns.
    Higher Sharpe = better risk-adjusted earnings quality.

    window : ~3 months of trading days
    """
    loc = returns.index.get_loc(date)

    if loc < window:
        return pd.Series(np.nan, index=returns.columns)

    ret_slice = returns.iloc[loc - window: loc]
    mean_ret  = ret_slice.mean()
    std_ret   = ret_slice.std(ddof=1)

    quality = mean_ret / std_ret.replace(0, np.nan)
    return quality


def factor_low_volatility(returns: pd.DataFrame, date: pd.Timestamp,
                           window: int = 63) -> pd.Series:
    """
    Low-Volatility factor: negative of trailing realized volatility.
    Lower vol = higher factor score (more attractive).
    """
    loc = returns.index.get_loc(date)

    if loc < window:
        return pd.Series(np.nan, index=returns.columns)

    ret_slice = returns.iloc[loc - window: loc]
    realized_vol = ret_slice.std(ddof=1)

    low_vol = -realized_vol  # negate so higher = better
    return low_vol


def factor_value(close: pd.DataFrame, date: pd.Timestamp,
                 window: int = 252) -> pd.Series:
    """
    Value proxy: negative of price relative to its 252-day trailing high.
    Assets far below their 52-week high are "cheap" on this metric.

    This is a pure-price Value signal (no P/B data needed from yfinance).
    Range: score in (-1, 0] -- more negative = further below 52W high = cheaper.
    """
    loc = close.index.get_loc(date)

    if loc < window:
        return pd.Series(np.nan, index=close.columns)

    price_now  = close.iloc[loc]
    rolling_high = close.iloc[loc - window: loc].max()

    # Ratio of current price to 52-week high, then negate
    value = -(price_now / rolling_high)
    return value


# ---------------------------------------------------------------------------
# Winsorization & z-scoring
# ---------------------------------------------------------------------------

def winsorize_series(s: pd.Series, lower: float = 0.05,
                     upper: float = 0.95) -> pd.Series:
    """Clip to empirical quantiles to remove outliers."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def cross_section_zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score: (x - mean) / std across all assets."""
    mean = s.mean()
    std  = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def prepare_factor(raw: pd.Series) -> pd.Series:
    """Winsorize then z-score a single raw factor series."""
    clean = raw.dropna()
    if len(clean) < 3:
        return raw  # not enough data
    win   = winsorize_series(clean)
    zscored = cross_section_zscore(win)
    return zscored.reindex(raw.index)


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

FACTOR_NAMES = ["momentum", "quality", "low_vol", "value"]


def compute_factor_scores(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    date: pd.Timestamp,
    weights: np.ndarray,
) -> pd.DataFrame:
    """
    Compute all four factor z-scores at a given rebalance date and
    return a DataFrame with one column per factor plus a 'composite' column.

    Parameters
    ----------
    close   : full adjusted close panel (T x N)
    returns : daily log returns panel (T x N)
    date    : rebalance date (must exist in close.index)
    weights : array of 4 weights summing to 1.0, order = FACTOR_NAMES

    Returns
    -------
    scores  : DataFrame with columns [momentum, quality, low_vol, value, composite]
              indexed by ticker
    """
    assert len(weights) == 4, "Provide exactly 4 factor weights"
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()  # normalise to sum-to-one

    # Compute raw factor series
    raw_mom  = factor_momentum(close, date)
    raw_qual = factor_quality(returns, date)
    raw_lvol = factor_low_volatility(returns, date)
    raw_val  = factor_value(close, date)

    # Winsorize + z-score each factor
    z_mom  = prepare_factor(raw_mom)
    z_qual = prepare_factor(raw_qual)
    z_lvol = prepare_factor(raw_lvol)
    z_val  = prepare_factor(raw_val)

    scores = pd.DataFrame({
        "momentum": z_mom,
        "quality":  z_qual,
        "low_vol":  z_lvol,
        "value":    z_val,
    })

    # Composite score
    factor_matrix = scores[FACTOR_NAMES].values
    scores["composite"] = factor_matrix @ weights

    return scores


# ---------------------------------------------------------------------------
# Factor attribution helper
# ---------------------------------------------------------------------------

def compute_factor_attribution(
    holdings_history: list[tuple[pd.Timestamp, list[str]]],
    scores_history:   list[pd.DataFrame],
    weights_history:  list[np.ndarray],
) -> pd.DataFrame:
    """
    Approximate factor attribution: for each rebalance period, compute
    the average factor z-score of the selected portfolio assets.

    Returns a DataFrame (periods x factors) showing which factors drove
    portfolio construction in each window.
    """
    records = []
    for (date, tickers), scores, weights in zip(
        holdings_history, scores_history, weights_history
    ):
        if scores is None or len(tickers) == 0:
            continue
        subset = scores.loc[scores.index.isin(tickers), FACTOR_NAMES]
        row = subset.mean()
        row["date"] = date
        row["regime_weights"] = str(np.round(weights, 3))
        records.append(row)

    if not records:
        return pd.DataFrame()

    attr = pd.DataFrame(records).set_index("date")
    return attr
