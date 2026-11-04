"""
metrics.py
----------
Performance analytics for the backtest results.

Computes a comprehensive set of risk/return statistics:
  - CAGR         : Compound Annual Growth Rate
  - Sharpe Ratio : annualized (252 trading days), risk-free = 0
  - Sortino Ratio: downside deviation-adjusted Sharpe
  - Max Drawdown : peak-to-trough decline in cumulative wealth
  - Calmar Ratio : CAGR / abs(Max Drawdown)
  - Win Rate     : fraction of monthly periods with positive return
  - Hit Ratio    : fraction of months the strategy outperformed SPY

All ratios assume 252 trading days per year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cagr(daily_returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (final_value / initial_value) ^ (252 / n_days) - 1
    """
    n = len(daily_returns)
    if n < 2:
        return np.nan
    cum = (1 + daily_returns).prod()
    years = n / TRADING_DAYS_PER_YEAR
    return cum ** (1 / years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    """
    Annualized Sharpe Ratio.
    Sharpe = (mean_daily_excess_return * 252) / (std_daily * sqrt(252))
    """
    excess = daily_returns - risk_free / TRADING_DAYS_PER_YEAR
    if excess.std(ddof=1) < 1e-10:
        return np.nan
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns: pd.Series, risk_free: float = 0.0,
                  mar: float = 0.0) -> float:
    """
    Annualized Sortino Ratio.
    Uses downside deviation (semi-standard deviation below MAR = 0).
    """
    excess = daily_returns - risk_free / TRADING_DAYS_PER_YEAR
    downside = daily_returns[daily_returns < mar]
    if len(downside) < 2:
        return np.nan
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_std < 1e-10:
        return np.nan
    return excess.mean() * TRADING_DAYS_PER_YEAR / downside_std


def max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of the cumulative return series.
    Returns a negative value (e.g., -0.25 = 25% drawdown).
    """
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    return float(drawdowns.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    """Calmar = CAGR / abs(Max Drawdown)."""
    mdd = abs(max_drawdown(daily_returns))
    if mdd < 1e-10:
        return np.nan
    return cagr(daily_returns) / mdd


def win_rate(daily_returns: pd.Series, freq: str = "ME") -> float:
    """
    Fraction of resampled periods (monthly by default) with positive returns.
    """
    monthly = daily_returns.resample(freq).sum()
    if len(monthly) == 0:
        return np.nan
    return (monthly > 0).mean()


def hit_ratio(strategy_returns: pd.Series,
              benchmark_returns: pd.Series,
              freq: str = "ME") -> float:
    """
    Fraction of months where strategy outperforms benchmark.
    """
    strat_m = strategy_returns.resample(freq).sum()
    bench_m = benchmark_returns.resample(freq).sum()
    aligned = pd.concat([strat_m, bench_m], axis=1).dropna()
    if len(aligned) == 0:
        return np.nan
    return (aligned.iloc[:, 0] > aligned.iloc[:, 1]).mean()


def information_ratio(strategy_returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
    """
    IR = annualized mean of excess daily returns / tracking error.
    """
    excess = strategy_returns - benchmark_returns.reindex(strategy_returns.index).fillna(0)
    if excess.std(ddof=1) < 1e-10:
        return np.nan
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Return the full time-series of drawdown from peak (negative values)."""
    cumulative  = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    return cumulative / rolling_max - 1


def cumulative_return_series(daily_returns: pd.Series) -> pd.Series:
    """Return the cumulative growth-of-$1 series."""
    return (1 + daily_returns).cumprod()


# ---------------------------------------------------------------------------
# Comprehensive summary table
# ---------------------------------------------------------------------------

def compute_metrics(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
    label: str = "Strategy",
) -> pd.DataFrame:
    """
    Compute all metrics for strategy and benchmark side-by-side.

    Returns a DataFrame with two columns (Strategy, Benchmark) and
    one row per metric.
    """
    bench_aligned = benchmark_returns.reindex(strategy_returns.index).fillna(0.0)

    def _metrics(rets: pd.Series) -> dict:
        return {
            "CAGR (%)":          round(cagr(rets) * 100, 2),
            "Ann. Vol (%)":      round(annualized_volatility(rets) * 100, 2),
            "Sharpe Ratio":      round(sharpe_ratio(rets), 3),
            "Sortino Ratio":     round(sortino_ratio(rets), 3),
            "Max Drawdown (%)":  round(max_drawdown(rets) * 100, 2),
            "Calmar Ratio":      round(calmar_ratio(rets), 3),
            "Win Rate (%)":      round(win_rate(rets) * 100, 2),
        }

    strat_m = _metrics(strategy_returns)
    bench_m = _metrics(bench_aligned)

    # Add IR and hit ratio (require both series)
    strat_m["Hit Ratio (%)"]  = round(hit_ratio(strategy_returns, bench_aligned) * 100, 2)
    strat_m["Info Ratio"]     = round(information_ratio(strategy_returns, bench_aligned), 3)
    bench_m["Hit Ratio (%)"]  = "--"
    bench_m["Info Ratio"]     = "--"

    df = pd.DataFrame({label: strat_m, "SPY Benchmark": bench_m})
    return df


# ---------------------------------------------------------------------------
# IS vs OOS comparison
# ---------------------------------------------------------------------------

def is_oos_comparison(
    is_returns:  pd.Series,
    oos_returns: pd.Series,
) -> pd.DataFrame:
    """
    Compare in-sample vs out-of-sample performance.
    A strategy with good OOS stats relative to IS is not overfit.
    """
    metrics_is  = {
        "CAGR (%)":         round(cagr(is_returns) * 100, 2)   if len(is_returns) > 10  else np.nan,
        "Sharpe Ratio":     round(sharpe_ratio(is_returns), 3) if len(is_returns) > 10  else np.nan,
        "Max Drawdown (%)": round(max_drawdown(is_returns) * 100, 2) if len(is_returns) > 10 else np.nan,
    }
    metrics_oos = {
        "CAGR (%)":         round(cagr(oos_returns) * 100, 2)   if len(oos_returns) > 10 else np.nan,
        "Sharpe Ratio":     round(sharpe_ratio(oos_returns), 3) if len(oos_returns) > 10 else np.nan,
        "Max Drawdown (%)": round(max_drawdown(oos_returns) * 100, 2) if len(oos_returns) > 10 else np.nan,
    }
    return pd.DataFrame({"In-Sample": metrics_is, "Out-of-Sample": metrics_oos})


# ---------------------------------------------------------------------------
# OOS window summary
# ---------------------------------------------------------------------------

def summarize_oos_windows(oos_windows: list[dict]) -> pd.DataFrame:
    """
    Return a tidy DataFrame summarizing each OOS test window.
    """
    records = []
    for w in oos_windows:
        records.append({
            "Window":       w["window"],
            "Test Start":   str(w["test_start"].date()),
            "Test End":     str(w["test_end"].date()),
            "OOS Return (%)": round(w["oos_return"] * 100, 2),
            "Rebalances":   w["n_rebalances"],
        })
    return pd.DataFrame(records).set_index("Window")
