"""
portfolio.py
------------
Constructs the long/short factor portfolio and applies beta-neutral hedging.

Rules
-----
- Rank all stocks by composite factor score.
- Long top 8, short bottom 8 (equal weight within each leg).
- Hedge residual net market beta with SPY (cash position).
- Output: dict of {ticker: weight} including SPY hedge weight.

Weights are fraction of total equity (long leg sums to +1, short leg to -1,
SPY hedge adds a small overlay). The combined portfolio is dollar-neutral and
approximately beta-neutral.
"""

import numpy as np
import pandas as pd
from typing import Optional


N_LONG  = 8
N_SHORT = 8


# ------------------------------------------------------------------ #
#  Core portfolio construction                                        #
# ------------------------------------------------------------------ #

def build_portfolio(
    scores: pd.Series,
    betas: pd.Series,
    n_long: int = N_LONG,
    n_short: int = N_SHORT,
    spy_ticker: str = "SPY",
) -> dict[str, float]:
    """
    Build an equal-weight long/short portfolio with SPY beta hedge.

    Parameters
    ----------
    scores     : cross-sectional factor scores indexed by ticker
    betas      : Kalman beta estimates indexed by ticker (same universe)
    n_long     : number of long positions
    n_short    : number of short positions
    spy_ticker : name of the SPY hedge instrument in the output dict

    Returns
    -------
    weights : dict {ticker: weight}
        Weights are fractions of total notional equity.
        Long leg: each stock weight = +1/n_long
        Short leg: each stock weight = -1/n_short
        SPY hedge: weight = -net_beta (so portfolio beta -> 0)
    """
    clean_scores = scores.dropna()
    if len(clean_scores) < n_long + n_short:
        # Not enough stocks — return empty
        return {}

    ranked = clean_scores.sort_values(ascending=False)
    long_tickers  = ranked.index[:n_long].tolist()
    short_tickers = ranked.index[-n_short:].tolist()

    weights = {}
    for t in long_tickers:
        weights[t] = 1.0 / n_long
    for t in short_tickers:
        weights[t] = -1.0 / n_short

    # Beta neutralization
    net_beta = _compute_net_beta(weights, betas)
    weights[spy_ticker] = -net_beta   # hedge: short beta * SPY to offset

    return weights


def _compute_net_beta(weights: dict, betas: pd.Series) -> float:
    """
    Sum of (weight_i * beta_i) for all non-SPY positions.
    """
    net = 0.0
    for ticker, w in weights.items():
        if ticker == "SPY":
            continue
        b = betas.get(ticker, np.nan)
        if not np.isnan(b):
            net += w * b
    return net


# ------------------------------------------------------------------ #
#  Turnover / transaction cost helpers                                #
# ------------------------------------------------------------------ #

def compute_turnover(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    """
    One-way portfolio turnover between two weight dictionaries.
    turnover = 0.5 * sum(|w_new - w_old|)
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    total = 0.0
    for t in all_tickers:
        diff = new_weights.get(t, 0.0) - old_weights.get(t, 0.0)
        total += abs(diff)
    return 0.5 * total


def apply_transaction_costs(
    gross_return: float,
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    cost_bps: float = 10.0,
) -> float:
    """
    Subtract transaction costs from a gross period return.

    cost_bps : one-way cost in basis points (default 10 bps = 0.10%)
    """
    turnover = compute_turnover(old_weights, new_weights)
    cost = turnover * (cost_bps / 10_000.0) * 2   # round-trip: 2 x one-way
    return gross_return - cost


# ------------------------------------------------------------------ #
#  Portfolio return computation                                       #
# ------------------------------------------------------------------ #

def compute_portfolio_return(
    weights: dict[str, float],
    daily_returns: pd.Series,
    spy_return: float,
) -> float:
    """
    Compute one-period portfolio return given current weights.

    Parameters
    ----------
    weights      : {ticker: weight} including optional SPY hedge
    daily_returns: Series of stock returns indexed by ticker
    spy_return   : SPY return for this period

    Returns
    -------
    Weighted sum of returns (float).
    """
    total = 0.0
    for ticker, w in weights.items():
        if ticker == "SPY":
            total += w * spy_return
        else:
            r = daily_returns.get(ticker, np.nan)
            if not np.isnan(r):
                total += w * r
    return total
