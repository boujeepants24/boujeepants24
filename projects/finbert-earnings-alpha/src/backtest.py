"""
backtest.py
-----------
Event-driven backtester for the FinBERT earnings sentiment strategy.

Strategy mechanics:
  - Universe: 30 large-cap US stocks.
  - Trigger: Earnings announcement date (simulated quarterly).
  - Entry: Market-close of the announcement day.
  - Signal: FinBERT composite sentiment score.
    * Long top-5 positive sentiment stocks.
    * Short bottom-5 negative sentiment stocks.
  - Exit: Market-close 5 trading days after entry.
  - Position sizing: Equal-weight within long and short books.
    Each leg (long or short) is sized to represent 50% of portfolio NAV.
  - Transaction costs: 10 basis points (0.10%) per trade (one-way),
    applied at both entry and exit.
  - Portfolio NAV starts at $1,000,000.

Performance metrics tracked per trade:
  gross_pnl, net_pnl, entry_price, exit_price, trade_return,
  sentiment_score, signal (1=long, -1=short)

Portfolio equity curve is built from the time-series of daily NAV.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_NAV       = 1_000_000.0   # Starting portfolio value
TRANSACTION_COST  = 0.001         # 10 bps per trade, one-way
LONG_ALLOCATION   = 0.50          # Fraction of NAV allocated to long book
SHORT_ALLOCATION  = 0.50          # Fraction of NAV allocated to short book
HOLD_DAYS         = 5             # Trading days to hold position


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

class Trade:
    """Immutable record of a single event-driven trade."""

    __slots__ = [
        "ticker", "earnings_date", "entry_date", "exit_date",
        "signal", "entry_price", "exit_price",
        "composite_score", "sentiment_label",
        "gross_return", "cost_return", "net_return",
        "gross_pnl", "net_pnl", "position_notional",
        "car_m1_p5",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {s: getattr(self, s, None) for s in self.__slots__}


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

def run_backtest(
    master_df: pd.DataFrame,
    prices: pd.DataFrame,
    initial_nav: float = INITIAL_NAV,
    transaction_cost: float = TRANSACTION_COST,
    long_allocation: float = LONG_ALLOCATION,
    short_allocation: float = SHORT_ALLOCATION,
    hold_days: int = HOLD_DAYS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the event-driven backtest.

    Parameters
    ----------
    master_df         : pd.DataFrame
        Output of signals.merge_signals_with_events(). Contains one row
        per earnings event with signal (1, -1, 0) and position_size.
    prices            : pd.DataFrame
        Adjusted close prices; index=dates, columns=tickers.
    initial_nav       : float  Starting portfolio value.
    transaction_cost  : float  One-way cost as fraction of trade value.
    long_allocation   : float  Fraction of NAV for long book.
    short_allocation  : float  Fraction of NAV for short book.
    hold_days         : int    Trading days to hold each position.
    verbose           : bool   Print progress.

    Returns
    -------
    trades_df   : pd.DataFrame  One row per trade with full P&L details.
    equity_df   : pd.DataFrame  Daily portfolio NAV (equity curve).
    """
    trading_days = pd.DatetimeIndex(prices.index).sort_values()
    nav          = initial_nav
    trades       = []

    # Only process events with a live signal
    active_events = master_df.loc[master_df["signal"] != 0].copy()
    active_events  = active_events.sort_values("earnings_date").reset_index(drop=True)

    total_events = len(active_events)
    if verbose:
        print(f"[backtest] Processing {total_events} active trade signals ...")

    # Group events by batch_id to apply portfolio-level sizing
    # Events in same batch share the NAV at time of entry
    batches = active_events.groupby("batch_id")

    for batch_id, batch in batches:
        batch_date = batch["earnings_date"].iloc[0]

        # Current position size allocation
        long_signals  = batch[batch["signal"] == 1]
        short_signals = batch[batch["signal"] == -1]

        n_long  = len(long_signals)
        n_short = len(short_signals)

        # Process each trade in the batch
        for _, row in batch.iterrows():
            ticker      = row["ticker"]
            entry_date  = row["earnings_date"]
            signal      = int(row["signal"])
            score       = float(row["composite_score"])
            pos_size    = float(row["position_size"])
            car_val     = row.get("car_m1_p5", np.nan)

            # --- Find entry price ---
            if entry_date not in prices.index or ticker not in prices.columns:
                continue
            entry_price = float(prices.loc[entry_date, ticker])
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            # --- Find exit date (HOLD_DAYS trading days after entry) ---
            entry_pos   = trading_days.searchsorted(entry_date)
            exit_pos    = min(entry_pos + hold_days, len(trading_days) - 1)
            exit_date   = trading_days[exit_pos]

            if exit_date not in prices.index or ticker not in prices.columns:
                continue
            exit_price = float(prices.loc[exit_date, ticker])
            if np.isnan(exit_price) or exit_price <= 0:
                continue

            # --- Compute returns ---
            # Gross return (log return, direction adjusted for signal)
            log_ret    = np.log(exit_price / entry_price)
            gross_ret  = signal * log_ret  # positive for correct direction

            # Transaction cost applies at entry and exit (two-way)
            cost_ret   = -2.0 * transaction_cost
            net_ret    = gross_ret + cost_ret

            # --- Compute P&L ---
            if signal == 1:
                # Long: allocate fraction of long book
                notional = nav * long_allocation * pos_size
            else:
                # Short: allocate fraction of short book
                notional = nav * short_allocation * pos_size

            gross_pnl = notional * (np.exp(gross_ret) - 1.0)
            net_pnl   = notional * (np.exp(net_ret)   - 1.0)

            # Determine sentiment label
            if score > 0.1:
                sentiment_label = "positive"
            elif score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            trade = Trade(
                ticker            = ticker,
                earnings_date     = entry_date,
                entry_date        = entry_date,
                exit_date         = exit_date,
                signal            = signal,
                entry_price       = entry_price,
                exit_price        = exit_price,
                composite_score   = score,
                sentiment_label   = sentiment_label,
                gross_return      = gross_ret,
                cost_return       = cost_ret,
                net_return        = net_ret,
                gross_pnl         = gross_pnl,
                net_pnl           = net_pnl,
                position_notional = notional,
                car_m1_p5         = float(car_val) if pd.notna(car_val) else np.nan,
            )
            trades.append(trade)

    # Build trades DataFrame
    trades_df = pd.DataFrame([t.to_dict() for t in trades])

    if len(trades_df) == 0:
        print("[backtest] WARNING: No trades generated.")
        return trades_df, pd.DataFrame()

    # Sort trades by entry date
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    # --- Build equity curve ---
    equity_df = _build_equity_curve(trades_df, trading_days, initial_nav)

    if verbose:
        total_pnl = trades_df["net_pnl"].sum()
        n_trades  = len(trades_df)
        win_rate  = (trades_df["net_pnl"] > 0).mean()
        print(f"[backtest] Completed: {n_trades} trades, total net P&L: ${total_pnl:,.0f}, "
              f"win rate: {win_rate:.1%}")

    return trades_df, equity_df


# ---------------------------------------------------------------------------
# Equity curve builder
# ---------------------------------------------------------------------------

def _build_equity_curve(
    trades_df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    initial_nav: float,
) -> pd.DataFrame:
    """
    Build the daily portfolio equity curve from trade P&L.

    Each trade contributes its net_pnl on the exit date.
    The equity curve accumulates from initial_nav.

    Parameters
    ----------
    trades_df    : pd.DataFrame  Trade records with exit_date and net_pnl.
    trading_days : DatetimeIndex All trading days in the period.
    initial_nav  : float         Starting portfolio value.

    Returns
    -------
    pd.DataFrame with columns: date, nav, daily_return, cumulative_return.
    """
    # Sum net_pnl by exit date
    pnl_by_date = (
        trades_df.groupby("exit_date")["net_pnl"]
        .sum()
        .reindex(trading_days, fill_value=0.0)
    )

    # Build cumulative NAV
    nav_series = initial_nav + pnl_by_date.cumsum()

    equity_df = pd.DataFrame({
        "date":       pnl_by_date.index,
        "daily_pnl":  pnl_by_date.values,
        "nav":        nav_series.values,
    })
    equity_df["daily_return"]      = equity_df["nav"].pct_change().fillna(0.0)
    equity_df["cumulative_return"] = equity_df["nav"] / initial_nav - 1.0

    return equity_df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def backtest_summary(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Compute portfolio-level performance statistics.

    Parameters
    ----------
    trades_df             : pd.DataFrame  Trade records.
    equity_df             : pd.DataFrame  Daily equity curve.
    risk_free_rate        : float         Annual risk-free rate (for Sharpe).
    trading_days_per_year : int

    Returns
    -------
    dict with performance metrics.
    """
    if len(trades_df) == 0 or len(equity_df) == 0:
        return {}

    daily_rets = equity_df["daily_return"].values
    nav        = equity_df["nav"].values
    net_pnls   = trades_df["net_pnl"].values

    # Total return
    total_ret = (nav[-1] / nav[0]) - 1.0

    # Annualised return (CAGR)
    n_days    = len(daily_rets)
    n_years   = n_days / trading_days_per_year
    cagr      = (nav[-1] / nav[0]) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # Volatility (annualised)
    vol_ann   = np.nanstd(daily_rets, ddof=1) * np.sqrt(trading_days_per_year)

    # Sharpe ratio
    daily_rf  = risk_free_rate / trading_days_per_year
    excess    = daily_rets - daily_rf
    sharpe    = (np.nanmean(excess) / np.nanstd(excess, ddof=1) * np.sqrt(trading_days_per_year)
                 if np.nanstd(excess, ddof=1) > 0 else 0.0)

    # Maximum drawdown
    cummax    = np.maximum.accumulate(nav)
    drawdowns = (nav - cummax) / cummax
    max_dd    = float(drawdowns.min())

    # Trade-level stats
    n_trades      = len(trades_df)
    n_long        = int((trades_df["signal"] == 1).sum())
    n_short       = int((trades_df["signal"] == -1).sum())
    win_rate      = float((net_pnls > 0).mean())
    avg_win       = float(net_pnls[net_pnls > 0].mean()) if (net_pnls > 0).any() else 0.0
    avg_loss      = float(net_pnls[net_pnls < 0].mean()) if (net_pnls < 0).any() else 0.0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    total_pnl     = float(net_pnls.sum())
    avg_pnl       = float(np.mean(net_pnls))

    return {
        "total_return_%":    round(total_ret * 100, 2),
        "cagr_%":            round(cagr * 100, 2),
        "volatility_ann_%":  round(vol_ann * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "max_drawdown_%":    round(max_dd * 100, 2),
        "n_trades":          n_trades,
        "n_long":            n_long,
        "n_short":           n_short,
        "win_rate_%":        round(win_rate * 100, 2),
        "avg_win_$":         round(avg_win, 2),
        "avg_loss_$":        round(avg_loss, 2),
        "profit_factor":     round(profit_factor, 3),
        "total_net_pnl_$":   round(total_pnl, 2),
        "avg_pnl_per_trade_$": round(avg_pnl, 2),
    }
