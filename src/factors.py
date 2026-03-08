"""
factors.py
----------
Computes cross-sectional factor scores used to rank stocks each month.

Factors
-------
1. Momentum  : 12-1 month return (skip last 21 days to avoid short-term reversal).
               Uses cumulative log return from day -252 to day -21.
2. Quality   : Rolling 63-day Sharpe ratio of daily returns (consistent performance proxy).

Composite score = 0.5 * z(momentum) + 0.5 * z(quality)

All z-scores are cross-sectional (across stocks) at each rebalance date.
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Individual factor computation                                       #
# ------------------------------------------------------------------ #

def compute_momentum(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """
    Cumulative log return from [t - lookback] to [t - skip].

    Parameters
    ----------
    returns  : DataFrame of daily log returns, shape (T, N)
    lookback : formation window in trading days  (default 252 ~ 12 months)
    skip     : recent days to skip               (default  21 ~  1 month)

    Returns
    -------
    pd.Series indexed by ticker with raw momentum score at the current date.
    """
    if len(returns) < lookback:
        return pd.Series(np.nan, index=returns.columns)

    window = returns.iloc[-(lookback): -skip] if skip > 0 else returns.iloc[-lookback:]
    momentum = window.sum(axis=0)   # sum of log returns = cumulative log return
    return momentum


def compute_quality(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Rolling Sharpe ratio over the last `window` trading days.

    Annualised = mean(r) / std(r) * sqrt(252).

    Returns
    -------
    pd.Series indexed by ticker.
    """
    if len(returns) < window:
        return pd.Series(np.nan, index=returns.columns)

    recent = returns.iloc[-window:]
    mean_r = recent.mean(axis=0)
    std_r  = recent.std(axis=0, ddof=1)
    sharpe = mean_r / std_r.replace(0, np.nan) * np.sqrt(252)
    return sharpe


# ------------------------------------------------------------------ #
#  Cross-sectional z-score                                            #
# ------------------------------------------------------------------ #

def cross_section_zscore(series: pd.Series) -> pd.Series:
    """Demean and scale a cross-sectional series to z-scores. Drops NaN before computing."""
    clean = series.dropna()
    if clean.std() == 0:
        return pd.Series(0.0, index=series.index)
    z = (clean - clean.mean()) / clean.std(ddof=1)
    return z.reindex(series.index)


# ------------------------------------------------------------------ #
#  Composite factor score at a single rebalance date                  #
# ------------------------------------------------------------------ #

def compute_factor_scores(
    returns: pd.DataFrame,
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
    quality_window: int = 63,
    w_momentum: float = 0.5,
    w_quality: float = 0.5,
) -> pd.Series:
    """
    Compute composite factor scores for all tickers given returns up to (and
    including) the rebalance date.

    Returns
    -------
    pd.Series of composite scores, indexed by ticker.
    NaN entries are excluded from ranking downstream.
    """
    mom_raw = compute_momentum(returns, lookback=momentum_lookback, skip=momentum_skip)
    qua_raw = compute_quality(returns, window=quality_window)

    mom_z = cross_section_zscore(mom_raw)
    qua_z = cross_section_zscore(qua_raw)

    # Align and combine
    combined = w_momentum * mom_z + w_quality * qua_z
    return combined


# ------------------------------------------------------------------ #
#  Batch computation over all rebalance dates (for analysis)          #
# ------------------------------------------------------------------ #

def compute_all_scores(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute factor scores at every rebalance date.

    Returns
    -------
    pd.DataFrame shape (n_rebalance_dates, n_tickers)
    """
    scores = {}
    for dt in rebalance_dates:
        hist = returns.loc[:dt]
        scores[dt] = compute_factor_scores(hist, **kwargs)
    return pd.DataFrame(scores).T
