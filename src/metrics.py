"""
metrics.py
----------
Performance analytics and P&L attribution for the funding arb strategy.

Metrics computed
----------------
  Sharpe ratio    : annualized (8h periods -> * sqrt(3*365))
  Sortino ratio   : same annualization, downside deviation in denominator
  Max Drawdown    : from peak NAV to trough, in percent
  CAGR            : compound annual growth rate
  Calmar ratio    : CAGR / |Max Drawdown|
  Funding yield   : total funding income / average NAV, annualized
  Win rate        : % of 8h periods with positive P&L
  Profit factor   : sum of gains / sum of losses

P&L Attribution (monthly)
--------------------------
  funding_income  : all 8h funding payments received
  basis_pnl       : mark-to-market changes in basis spread
  fee_cost        : taker fees + slippage
  borrow_cost     : carry cost on spot leg
  net_pnl         : sum of all four
"""

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIODS_PER_YEAR = 3 * 365   # 3 funding periods per day


# ---------------------------------------------------------------------------
# Return series
# ---------------------------------------------------------------------------

def compute_period_returns(nav_series: pd.Series) -> pd.Series:
    """
    Compute 8h percentage returns from NAV series.

    Returns
    -------
    pd.Series of fractional returns, same index as nav_series.
    """
    r = nav_series.pct_change().dropna()
    r.name = "period_return"
    return r


def compute_dollar_returns(ledger: pd.DataFrame) -> pd.Series:
    """
    Period-level net P&L in USD from the ledger.
    = period_funding + period_basis_pnl - period_fees - period_borrow
    """
    pnl = (
        ledger["period_funding"]
        + ledger["period_basis_pnl"]
        - ledger["period_fees"]
        - ledger["period_borrow"]
    )
    pnl.name = "period_pnl_usd"
    return pnl


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, rf_per_period: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    returns          : 8h fractional returns
    rf_per_period    : risk-free rate per 8h period (default 0)
    """
    excess = returns - rf_per_period
    if excess.std() == 0:
        return 0.0
    sr = excess.mean() / excess.std() * np.sqrt(PERIODS_PER_YEAR)
    return float(sr)


def sortino_ratio(returns: pd.Series, rf_per_period: float = 0.0) -> float:
    """
    Annualized Sortino ratio (downside deviation in denominator).
    """
    excess    = returns - rf_per_period
    downside  = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf")
    sortino = excess.mean() / downside.std() * np.sqrt(PERIODS_PER_YEAR)
    return float(sortino)


def max_drawdown(nav_series: pd.Series) -> float:
    """
    Maximum drawdown: largest peak-to-trough decline, expressed as a positive fraction.

    E.g., 0.15 means the portfolio fell 15% from its previous peak.
    """
    rolling_max = nav_series.cummax()
    drawdown    = (nav_series - rolling_max) / rolling_max
    return float(drawdown.min())   # most negative value


def drawdown_series(nav_series: pd.Series) -> pd.Series:
    """Return the full drawdown series (negative fractions) for plotting."""
    rolling_max = nav_series.cummax()
    dd          = (nav_series - rolling_max) / rolling_max
    dd.name     = "drawdown"
    return dd


def cagr(nav_series: pd.Series) -> float:
    """
    Compound Annual Growth Rate over the full backtest period.

    CAGR = (final_nav / initial_nav) ^ (periods_per_year / n_periods) - 1
    """
    n_periods  = len(nav_series)
    if n_periods < 2:
        return 0.0
    ratio      = nav_series.iloc[-1] / nav_series.iloc[0]
    if ratio <= 0:
        return -1.0
    years      = n_periods / PERIODS_PER_YEAR
    return float(ratio ** (1 / years) - 1)


def calmar_ratio(nav_series: pd.Series) -> float:
    """Calmar = CAGR / |Max Drawdown|"""
    mdd = abs(max_drawdown(nav_series))
    if mdd == 0:
        return float("inf")
    return float(cagr(nav_series) / mdd)


def funding_yield(ledger: pd.DataFrame, nav_series: pd.Series) -> float:
    """
    Annualized funding yield = total_funding_income / avg_nav.
    """
    total_funding = ledger["period_funding"].sum()
    avg_nav       = nav_series.mean()
    n_periods     = len(ledger)
    if avg_nav == 0 or n_periods == 0:
        return 0.0
    periods_in_data = n_periods
    ann_factor      = PERIODS_PER_YEAR / periods_in_data
    return float(total_funding / avg_nav * ann_factor)


def win_rate(period_returns: pd.Series) -> float:
    """Fraction of periods with positive returns."""
    if len(period_returns) == 0:
        return 0.0
    return float((period_returns > 0).mean())


def profit_factor(dollar_pnl: pd.Series) -> float:
    """Sum of gains / sum of absolute losses."""
    gains  = dollar_pnl[dollar_pnl > 0].sum()
    losses = dollar_pnl[dollar_pnl < 0].abs().sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def skewness(returns: pd.Series) -> float:
    """Third standardized moment of return distribution."""
    return float(returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (Fisher definition) of return distribution."""
    return float(returns.kurtosis())


# ---------------------------------------------------------------------------
# P&L Attribution
# ---------------------------------------------------------------------------

def monthly_pnl_attribution(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-period attribution into monthly buckets.

    Returns
    -------
    DataFrame indexed by YYYY-MM (Period), columns:
      funding_income, basis_pnl, fee_cost, borrow_cost, net_pnl
    """
    df = ledger.copy()
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")

    monthly = df.groupby("month").agg(
        funding_income = ("period_funding",   "sum"),
        basis_pnl      = ("period_basis_pnl", "sum"),
        fee_cost       = ("period_fees",       "sum"),
        borrow_cost    = ("period_borrow",     "sum"),
    )
    monthly["net_pnl"] = (
        monthly["funding_income"]
        + monthly["basis_pnl"]
        - monthly["fee_cost"]
        - monthly["borrow_cost"]
    )
    return monthly


def quarterly_pnl_attribution(ledger: pd.DataFrame) -> pd.DataFrame:
    """Same as monthly but grouped by quarter."""
    df = ledger.copy()
    df.index = pd.to_datetime(df.index)
    df["quarter"] = df.index.to_period("Q")

    quarterly = df.groupby("quarter").agg(
        funding_income = ("period_funding",   "sum"),
        basis_pnl      = ("period_basis_pnl", "sum"),
        fee_cost       = ("period_fees",       "sum"),
        borrow_cost    = ("period_borrow",     "sum"),
    )
    quarterly["net_pnl"] = (
        quarterly["funding_income"]
        + quarterly["basis_pnl"]
        - quarterly["fee_cost"]
        - quarterly["borrow_cost"]
    )
    return quarterly


# ---------------------------------------------------------------------------
# Full performance summary
# ---------------------------------------------------------------------------

def performance_summary(
    ledger:     pd.DataFrame,
    initial_nav: float = 1_000_000.0,
) -> dict:
    """
    Compute and return a complete performance summary dictionary.

    Parameters
    ----------
    ledger      : per-period backtest ledger from backtest.py
    initial_nav : starting portfolio value in USD

    Returns
    -------
    dict of performance metrics (all scalars)
    """
    nav_series    = ledger["nav"]
    period_ret    = compute_period_returns(nav_series)
    dollar_pnl    = compute_dollar_returns(ledger)

    summary = {
        # Returns
        "total_return_pct":    (nav_series.iloc[-1] / initial_nav - 1) * 100,
        "cagr_pct":            cagr(nav_series) * 100,
        "funding_yield_pct":   funding_yield(ledger, nav_series) * 100,

        # Risk-adjusted
        "sharpe":              sharpe_ratio(period_ret),
        "sortino":             sortino_ratio(period_ret),
        "calmar":              calmar_ratio(nav_series),

        # Drawdown
        "max_drawdown_pct":    max_drawdown(nav_series) * 100,

        # Efficiency
        "win_rate_pct":        win_rate(period_ret) * 100,
        "profit_factor":       profit_factor(dollar_pnl),

        # Distribution
        "return_skewness":     skewness(period_ret),
        "return_kurtosis":     kurtosis(period_ret),

        # Attribution totals
        "total_funding_usd":   ledger["period_funding"].sum(),
        "total_basis_pnl_usd": ledger["period_basis_pnl"].sum(),
        "total_fees_usd":      ledger["period_fees"].sum(),
        "total_borrow_usd":    ledger["period_borrow"].sum(),
        "net_pnl_usd":         nav_series.iloc[-1] - initial_nav,

        # Exposure
        "avg_leverage":        ledger["leverage"].mean(),
        "max_leverage":        ledger["leverage"].max(),
        "avg_n_positions":     ledger["n_positions"].mean(),
        "max_n_positions":     ledger["n_positions"].max(),

        # Period info
        "n_periods":           len(ledger),
        "years":               len(ledger) / PERIODS_PER_YEAR,
        "initial_nav":         initial_nav,
        "final_nav":           nav_series.iloc[-1],
    }
    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print the performance summary to stdout (ASCII-safe)."""
    sep = "-" * 52
    print(sep)
    print("  FUNDING RATE ARB BACKTEST - PERFORMANCE SUMMARY")
    print(sep)
    print(f"  Period             : {summary['years']:.1f} years  ({summary['n_periods']} 8h bars)")
    print(f"  Initial NAV        : ${summary['initial_nav']:>14,.2f}")
    print(f"  Final NAV          : ${summary['final_nav']:>14,.2f}")
    print(f"  Net P&L            : ${summary['net_pnl_usd']:>+14,.2f}")
    print(sep)
    print(f"  Total Return       : {summary['total_return_pct']:>+8.2f}%")
    print(f"  CAGR               : {summary['cagr_pct']:>+8.2f}%")
    print(f"  Funding Yield (ann): {summary['funding_yield_pct']:>+8.2f}%")
    print(sep)
    print(f"  Sharpe Ratio       : {summary['sharpe']:>8.3f}")
    print(f"  Sortino Ratio      : {summary['sortino']:>8.3f}")
    print(f"  Calmar Ratio       : {summary['calmar']:>8.3f}")
    print(f"  Max Drawdown       : {summary['max_drawdown_pct']:>8.2f}%")
    print(sep)
    print(f"  Win Rate           : {summary['win_rate_pct']:>8.2f}%")
    print(f"  Profit Factor      : {summary['profit_factor']:>8.3f}")
    print(f"  Return Skewness    : {summary['return_skewness']:>8.3f}")
    print(f"  Return Kurtosis    : {summary['return_kurtosis']:>8.3f}")
    print(sep)
    print(f"  P&L Attribution:")
    print(f"    Funding Income   : ${summary['total_funding_usd']:>+14,.2f}")
    print(f"    Basis P&L        : ${summary['total_basis_pnl_usd']:>+14,.2f}")
    print(f"    Fee Cost         : ${summary['total_fees_usd']:>-14,.2f}")
    print(f"    Borrow Cost      : ${summary['total_borrow_usd']:>-14,.2f}")
    print(sep)
    print(f"  Avg Leverage       : {summary['avg_leverage']:>8.3f}x")
    print(f"  Max Leverage       : {summary['max_leverage']:>8.3f}x")
    print(f"  Avg Open Positions : {summary['avg_n_positions']:>8.2f}")
    print(sep)
