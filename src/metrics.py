"""
metrics.py
----------
Performance analytics for the Kalman Factor-Neutral strategy.

Computed metrics
----------------
CAGR         : Compound Annual Growth Rate
Sharpe       : Annualised Sharpe ratio (rf = 0)
Sortino      : Annualised Sortino ratio (rf = 0)
Max Drawdown : Peak-to-trough drawdown
Calmar       : CAGR / |Max Drawdown|
Volatility   : Annualised daily return std
Correlation  : Pearson correlation to SPY
Hit Rate     : Fraction of positive return months
"""

import numpy as np
import pandas as pd
from typing import Optional

# pandas 2.2+ uses "ME" for month-end; older versions use "M"
_PD_VER = tuple(int(x) for x in pd.__version__.split(".")[:2])
_MONTH_END_FREQ = "ME" if _PD_VER >= (2, 2) else "M"


TRADING_DAYS_PER_YEAR = 252


# ------------------------------------------------------------------ #
#  Individual metrics                                                 #
# ------------------------------------------------------------------ #

def cagr(daily_returns: pd.Series) -> float:
    """Compound Annual Growth Rate from a series of daily log returns."""
    total_log_return = daily_returns.sum()
    n_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return np.nan
    return np.exp(total_log_return / n_years) - 1.0


def annualised_volatility(daily_returns: pd.Series) -> float:
    """Annualised standard deviation of daily log returns."""
    return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio (rf denominated in daily terms)."""
    excess = daily_returns - rf / TRADING_DAYS_PER_YEAR
    if excess.std(ddof=1) == 0:
        return np.nan
    return excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sortino ratio using downside deviation."""
    excess = daily_returns - rf / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return np.nan
    downside_vol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (excess.mean() * TRADING_DAYS_PER_YEAR) / downside_vol


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    running_max = equity_curve.cummax()
    drawdown    = equity_curve / running_max - 1.0
    return drawdown.min()


def calmar_ratio(daily_returns: pd.Series, equity_curve: pd.Series) -> float:
    """CAGR divided by absolute max drawdown."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return np.nan
    return cagr(daily_returns) / abs(mdd)


def correlation_to_benchmark(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Pearson correlation between strategy and benchmark daily returns."""
    aligned = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the time series of drawdown values."""
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Aggregate daily log returns to a calendar month heatmap DataFrame.

    Returns
    -------
    DataFrame with years as index and months (1-12) as columns.
    Values are simple percentage returns (not log returns).
    """
    # Convert log returns to simple returns per month
    monthly_log = daily_returns.resample(_MONTH_END_FREQ).sum()
    monthly_simple = (np.exp(monthly_log) - 1.0) * 100.0   # in percent

    pivot = pd.DataFrame({
        "year" : monthly_simple.index.year,
        "month": monthly_simple.index.month,
        "ret"  : monthly_simple.values,
    })
    heatmap = pivot.pivot(index="year", columns="month", values="ret")
    heatmap.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    return heatmap


def rolling_sharpe(daily_returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    roll_mean = daily_returns.rolling(window).mean()
    roll_std  = daily_returns.rolling(window).std(ddof=1)
    return roll_mean / roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)


# ------------------------------------------------------------------ #
#  Composite summary                                                  #
# ------------------------------------------------------------------ #

def compute_all_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    spy_returns: pd.Series,
    label: str = "Strategy",
) -> dict:
    """
    Compute and return a dict of all performance metrics.
    """
    spy_aligned = spy_returns.reindex(daily_returns.index).dropna()
    ret_aligned = daily_returns.reindex(spy_aligned.index).dropna()

    metrics = {
        "label"        : label,
        "CAGR"         : cagr(daily_returns),
        "Volatility"   : annualised_volatility(daily_returns),
        "Sharpe"       : sharpe_ratio(daily_returns),
        "Sortino"      : sortino_ratio(daily_returns),
        "Max Drawdown" : max_drawdown(equity_curve),
        "Calmar"       : calmar_ratio(daily_returns, equity_curve),
        "Corr to SPY"  : correlation_to_benchmark(ret_aligned, spy_aligned),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict to stdout (ASCII-safe)."""
    print("")
    print("=" * 52)
    print(f"  Performance Summary: {metrics['label']}")
    print("=" * 52)
    print(f"  CAGR          : {metrics['CAGR']*100:>8.2f}%")
    print(f"  Volatility    : {metrics['Volatility']*100:>8.2f}%")
    print(f"  Sharpe Ratio  : {metrics['Sharpe']:>8.3f}")
    print(f"  Sortino Ratio : {metrics['Sortino']:>8.3f}")
    print(f"  Max Drawdown  : {metrics['Max Drawdown']*100:>8.2f}%")
    print(f"  Calmar Ratio  : {metrics['Calmar']:>8.3f}")
    print(f"  Corr to SPY   : {metrics['Corr to SPY']:>8.3f}")
    print("=" * 52)
    print("")


def print_comparison_table(metrics_list: list[dict]) -> None:
    """Print a side-by-side comparison table for multiple strategies."""
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Corr to SPY"]
    formats = {
        "CAGR"         : lambda v: f"{v*100:>8.2f}%",
        "Volatility"   : lambda v: f"{v*100:>8.2f}%",
        "Sharpe"       : lambda v: f"{v:>9.3f}",
        "Sortino"      : lambda v: f"{v:>9.3f}",
        "Max Drawdown" : lambda v: f"{v*100:>8.2f}%",
        "Calmar"       : lambda v: f"{v:>9.3f}",
        "Corr to SPY"  : lambda v: f"{v:>9.3f}",
    }
    labels = [m["label"] for m in metrics_list]
    col_w = max(12, max(len(l) for l in labels) + 2)

    header = f"{'Metric':<18}" + "".join(f"{l:>{col_w}}" for l in labels)
    print("")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for k in keys:
        row = f"{k:<18}"
        for m in metrics_list:
            v = m.get(k, np.nan)
            try:
                row += formats[k](v)
            except (TypeError, ValueError):
                row += f"{'N/A':>{col_w}}"
        print(row)
    print("=" * len(header))
    print("")
