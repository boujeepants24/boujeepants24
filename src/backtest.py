"""
backtest.py
-----------
Monthly rebalancing backtester for the Kalman Factor-Neutral strategy.

Simulation details
------------------
- Rebalances on the last trading day of each calendar month.
- Uses all returns up to (and including) the rebalance date for:
    * factor score computation
    * Kalman beta estimates (already pre-computed online)
- Between rebalance dates the portfolio is held at fixed weights.
- Transaction costs: 10 bps one-way (round-trip applied at each rebalance).
- Minimum warmup: 252 days before the first trade.
- First trade date: earliest month-end >= 252 days after the first data date.
"""

import numpy as np
import pandas as pd

from .factors import compute_factor_scores
from .kalman_beta import get_betas_at
from .portfolio import (
    build_portfolio,
    compute_portfolio_return,
    compute_turnover,
)


COST_BPS    = 10.0      # one-way transaction cost in basis points
WARMUP_DAYS = 252       # minimum trading days before first trade
SPY_TICKER  = "SPY"


# ------------------------------------------------------------------ #
#  Rebalance date generation                                          #
# ------------------------------------------------------------------ #

def get_month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day of each calendar month in `index`."""
    series = index.to_series()
    grouped = series.groupby([series.dt.year, series.dt.month]).last()
    return pd.DatetimeIndex(grouped.values)


# ------------------------------------------------------------------ #
#  Main backtest loop                                                 #
# ------------------------------------------------------------------ #

def run_backtest(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    beta_df: pd.DataFrame,
    cost_bps: float = COST_BPS,
    warmup: int = WARMUP_DAYS,
    n_long: int = 8,
    n_short: int = 8,
) -> dict:
    """
    Run the full monthly-rebalancing backtest.

    Parameters
    ----------
    returns     : (T, N) daily log returns for universe stocks
    spy_returns : (T,) SPY daily log returns
    beta_df     : (T, N) Kalman beta estimates (pre-computed)
    cost_bps    : one-way transaction cost in basis points
    warmup      : minimum trading days before first trade

    Returns
    -------
    dict with keys:
        equity_curve    : pd.Series  (portfolio equity, starting at 1.0)
        spy_equity      : pd.Series  (SPY equity, starting at 1.0)
        daily_returns   : pd.Series  (portfolio daily log returns)
        long_returns    : pd.Series  (long leg daily returns, attribution)
        short_returns   : pd.Series  (short leg daily returns, attribution)
        rebalance_log   : list of dicts, one per rebalance date
        weights_history : dict of {date: weights dict}
        rebalance_dates : DatetimeIndex of all rebalance dates
    """
    # Align indices
    common_idx = returns.index.intersection(spy_returns.index)
    ret = returns.loc[common_idx].copy()
    spy = spy_returns.loc[common_idx].copy()

    all_dates = ret.index

    # Rebalance schedule
    all_month_ends = get_month_end_dates(all_dates)

    # First tradable date: index[warmup] ensures 252 prior observations
    first_trade_date = all_dates[warmup]
    rebalance_dates  = all_month_ends[all_month_ends >= first_trade_date]

    print(f"Backtest period : {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}")
    print(f"Rebalance dates : {len(rebalance_dates)}")
    print("Running backtest ...")

    portfolio_returns: list[float]     = []
    dates_out:         list            = []
    rebalance_log:     list[dict]      = []
    weights_history:   dict            = {}

    current_weights: dict[str, float]  = {}
    rebalance_set = set(rebalance_dates)

    for i in range(len(all_dates)):
        date = all_dates[i]

        # ---- Rebalance if this is a scheduled rebalance date ----
        if date in rebalance_set:
            hist_returns = ret.iloc[: i + 1]           # all data up to today
            betas_today  = get_betas_at(beta_df, date)
            scores       = compute_factor_scores(hist_returns)

            new_weights = build_portfolio(
                scores,
                betas_today,
                n_long=n_long,
                n_short=n_short,
                spy_ticker=SPY_TICKER,
            )

            if new_weights:
                turnover = compute_turnover(current_weights, new_weights)
                cost     = turnover * (cost_bps / 10_000.0) * 2  # round-trip

                n_long_pos  = sum(1 for k, w in new_weights.items() if w > 0 and k != SPY_TICKER)
                n_short_pos = sum(1 for k, w in new_weights.items() if w < 0 and k != SPY_TICKER)

                rebalance_log.append({
                    "date"     : date,
                    "turnover" : turnover,
                    "cost"     : cost,
                    "n_long"   : n_long_pos,
                    "n_short"  : n_short_pos,
                    "spy_hedge": new_weights.get(SPY_TICKER, 0.0),
                })

                weights_history[date] = new_weights.copy()
                current_weights       = new_weights.copy()

        # ---- Compute daily return (skip day 0 — no prior weights) ----
        if current_weights and i > 0:
            gross_ret = compute_portfolio_return(
                current_weights,
                ret.iloc[i],
                spy.iloc[i],
            )

            # Deduct transaction cost on rebalance days
            if date in rebalance_set and rebalance_log:
                last_log = rebalance_log[-1]
                if last_log["date"] == date:
                    gross_ret -= last_log["cost"]

            portfolio_returns.append(gross_ret)
            dates_out.append(date)

    port_idx    = pd.DatetimeIndex(dates_out)
    port_series = pd.Series(portfolio_returns, index=port_idx, name="strategy")

    # ---- Equity curves (normalised to 1.0 at first trade date) ----
    equity_curve = np.exp(port_series.cumsum())
    equity_curve = equity_curve / equity_curve.iloc[0]

    spy_slice  = spy.loc[port_series.index]
    spy_equity = np.exp(spy_slice.cumsum())
    spy_equity = spy_equity / spy_equity.iloc[0]

    # ---- Attribution returns ----
    long_ret, short_ret = _attribution_returns(
        ret, spy, weights_history, port_idx
    )

    print(f"Backtest complete. {len(port_series)} daily return observations.")

    return {
        "equity_curve"   : equity_curve,
        "spy_equity"     : spy_equity,
        "daily_returns"  : port_series,
        "long_returns"   : long_ret,
        "short_returns"  : short_ret,
        "rebalance_log"  : rebalance_log,
        "weights_history": weights_history,
        "rebalance_dates": rebalance_dates,
    }


# ------------------------------------------------------------------ #
#  Long / short leg attribution                                       #
# ------------------------------------------------------------------ #

def _attribution_returns(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    weights_history: dict,
    trading_dates: pd.DatetimeIndex,
) -> tuple:
    """Compute separate daily return series for the long and short legs."""
    sorted_rebal = sorted(weights_history.keys())

    def get_leg_weights(date):
        """Return (long_weights, short_weights) active on `date`."""
        active = [d for d in sorted_rebal if d <= date]
        if not active:
            return {}, {}
        w = weights_history[active[-1]]
        long_w  = {k: v for k, v in w.items() if v > 0 and k != "SPY"}
        short_w = {k: v for k, v in w.items() if v < 0 and k != "SPY"}
        return long_w, short_w

    long_list  = []
    short_list = []

    for date in trading_dates:
        lw, sw = get_leg_weights(date)

        if date in returns.index:
            day_r = returns.loc[date]
        else:
            long_list.append(0.0)
            short_list.append(0.0)
            continue

        lr = sum(lw[t] * day_r.get(t, 0.0) for t in lw) if lw else 0.0
        sr = sum(sw[t] * day_r.get(t, 0.0) for t in sw) if sw else 0.0

        long_list.append(lr)
        short_list.append(sr)

    long_series  = pd.Series(long_list,  index=trading_dates, name="long_leg")
    short_series = pd.Series(short_list, index=trading_dates, name="short_leg")
    return long_series, short_series
