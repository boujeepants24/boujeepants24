"""
metrics.py
----------
Performance analytics and statistical metrics for the FinBERT alpha model.

Modules:
  1. Information Coefficient (IC) statistics — daily IC, IC IR, hit rate.
  2. CAR significance testing by sentiment bucket.
  3. Quintile return analysis (monotonicity test).
  4. Portfolio-level risk metrics: Sharpe, Sortino, Calmar, drawdown.
  5. Attribution: sentiment label breakdown of P&L.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# 1. Information Coefficient metrics
# ---------------------------------------------------------------------------

def compute_ic_timeseries(
    master_df: pd.DataFrame,
    score_col: str = "composite_score",
    return_col: str = "forward_5d_return",
    freq: str = "ME",   # month-end
) -> pd.DataFrame:
    """
    Compute monthly Information Coefficient (Spearman rank correlation)
    between composite_score and forward_5d_return.

    Parameters
    ----------
    master_df  : pd.DataFrame  One row per earnings event.
    score_col  : str           Sentiment score column.
    return_col : str           Forward return column.
    freq       : str           Pandas groupby frequency (default: month-end).

    Returns
    -------
    pd.DataFrame  Monthly IC values with columns: date, ic, n_obs.
    """
    df = master_df[[score_col, return_col, "earnings_date"]].dropna().copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.set_index("earnings_date").sort_index()

    records = []
    for period, group in df.resample(freq):
        if len(group) < 5:
            continue
        ic, p_val = stats.spearmanr(group[score_col], group[return_col])
        records.append({
            "date":  period,
            "ic":    float(ic),
            "n_obs": len(group),
            "p_val": float(p_val),
        })

    ic_df = pd.DataFrame(records).set_index("date")
    return ic_df


def ic_summary(ic_df: pd.DataFrame) -> dict:
    """
    Summarise IC time-series: mean IC, IC IR (information ratio), t-stat.

    IC Information Ratio = mean(IC) / std(IC)
    Annualised IC IR = IC IR * sqrt(12) for monthly IC.

    Parameters
    ----------
    ic_df : pd.DataFrame  Output of compute_ic_timeseries().

    Returns
    -------
    dict of IC statistics.
    """
    ic_vals = ic_df["ic"].dropna()
    n       = len(ic_vals)

    if n < 2:
        return {"mean_ic": np.nan, "ic_ir": np.nan}

    mean_ic  = float(ic_vals.mean())
    std_ic   = float(ic_vals.std(ddof=1))
    ic_ir    = mean_ic / std_ic if std_ic > 0 else 0.0
    ic_ir_ann = ic_ir * np.sqrt(12)  # annualise monthly
    t_stat   = float(stats.ttest_1samp(ic_vals, 0).statistic)
    p_value  = float(stats.ttest_1samp(ic_vals, 0).pvalue)
    hit_rate = float((ic_vals > 0).mean())

    return {
        "n_months":       n,
        "mean_ic":        round(mean_ic, 4),
        "std_ic":         round(std_ic, 4),
        "ic_ir":          round(ic_ir, 4),
        "ic_ir_ann":      round(ic_ir_ann, 4),
        "t_stat":         round(t_stat, 3),
        "p_value":        round(p_value, 4),
        "hit_rate":       round(hit_rate, 4),
        "significant_5pct": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# 2. CAR significance by sentiment bucket
# ---------------------------------------------------------------------------

def car_by_sentiment_bucket(
    master_df: pd.DataFrame,
    car_col: str = "car_m1_p5",
    sentiment_col: str = "sentiment_label",
) -> pd.DataFrame:
    """
    Compute mean CAR and t-statistics for each sentiment bucket.

    Parameters
    ----------
    master_df     : pd.DataFrame  Master signals + event study frame.
    car_col       : str           CAR column to analyse.
    sentiment_col : str           Column with 'positive'/'negative'/'neutral'.

    Returns
    -------
    pd.DataFrame  Summary table per sentiment bucket.
    """
    records = []
    for label in ["positive", "neutral", "negative"]:
        subset = master_df.loc[master_df[sentiment_col] == label, car_col].dropna()
        n      = len(subset)
        if n < 2:
            continue

        mean_car = float(subset.mean())
        std_car  = float(subset.std(ddof=1))
        t_stat   = float(stats.ttest_1samp(subset, 0).statistic)
        p_val    = float(stats.ttest_1samp(subset, 0).pvalue)

        records.append({
            "sentiment":   label,
            "n":           n,
            "mean_car":    round(mean_car, 5),
            "std_car":     round(std_car, 5),
            "t_stat":      round(t_stat, 3),
            "p_value":     round(p_val, 4),
            "sig_5pct":    p_val < 0.05,
        })

    return pd.DataFrame(records).set_index("sentiment")


# ---------------------------------------------------------------------------
# 3. Quintile return analysis
# ---------------------------------------------------------------------------

def quintile_returns_table(
    master_df: pd.DataFrame,
    return_col: str = "forward_5d_return",
    quintile_col: str = "quintile",
) -> pd.DataFrame:
    """
    Compute mean forward return by sentiment quintile.

    Q1 = most negative, Q5 = most positive.
    A monotonically increasing pattern validates the model.

    Returns
    -------
    pd.DataFrame  Mean return and t-stat per quintile.
    """
    records = []
    for q in range(1, 6):
        subset = master_df.loc[master_df[quintile_col] == q, return_col].dropna()
        n      = len(subset)
        if n < 2:
            continue
        mean_ret = float(subset.mean())
        std_ret  = float(subset.std(ddof=1))
        t_stat   = float(stats.ttest_1samp(subset, 0).statistic)
        p_val    = float(stats.ttest_1samp(subset, 0).pvalue)

        records.append({
            "quintile": q,
            "n":        n,
            "mean_fwd_return": round(mean_ret, 5),
            "std_fwd_return":  round(std_ret, 5),
            "t_stat":          round(t_stat, 3),
            "p_value":         round(p_val, 4),
        })

    df = pd.DataFrame(records).set_index("quintile")
    # Long-short spread: Q5 - Q1
    if len(df) >= 5:
        ls_spread = df.loc[5, "mean_fwd_return"] - df.loc[1, "mean_fwd_return"]
        print(f"[metrics] Q5-Q1 spread (forward 5d return): {ls_spread:.4f} ({ls_spread*100:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# 4. Portfolio risk metrics
# ---------------------------------------------------------------------------

def portfolio_risk_metrics(
    equity_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Compute comprehensive risk-adjusted performance metrics.

    Parameters
    ----------
    equity_df      : pd.DataFrame  Must have 'daily_return' and 'nav' columns.
    risk_free_rate : float         Annual risk-free rate.

    Returns
    -------
    dict with: cagr, volatility, sharpe, sortino, calmar, max_drawdown,
               max_dd_duration, var_95, cvar_95.
    """
    if len(equity_df) == 0:
        return {}

    rets = equity_df["daily_return"].values
    nav  = equity_df["nav"].values

    # Remove leading zeros before computing
    rets = rets[~np.isnan(rets)]

    n_days  = len(rets)
    n_years = n_days / trading_days_per_year

    # CAGR
    cagr    = (nav[-1] / nav[0]) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # Volatility
    vol_ann = np.std(rets, ddof=1) * np.sqrt(trading_days_per_year)

    # Sharpe
    daily_rf = risk_free_rate / trading_days_per_year
    excess   = rets - daily_rf
    sharpe   = (np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(trading_days_per_year)
                if np.std(excess, ddof=1) > 0 else 0.0)

    # Sortino (downside deviation)
    downside = excess[excess < 0]
    down_vol = np.std(downside, ddof=1) * np.sqrt(trading_days_per_year) if len(downside) > 1 else 1e-9
    sortino  = np.mean(excess) * trading_days_per_year / down_vol if down_vol > 0 else 0.0

    # Drawdown
    cummax        = np.maximum.accumulate(nav)
    drawdown_pct  = (nav - cummax) / cummax
    max_dd        = float(drawdown_pct.min())

    # Max drawdown duration (in trading days)
    in_dd         = drawdown_pct < 0
    dd_duration   = 0
    current_dur   = 0
    for flag in in_dd:
        if flag:
            current_dur += 1
            dd_duration = max(dd_duration, current_dur)
        else:
            current_dur = 0

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # VaR and CVaR at 95%
    var_95  = float(np.percentile(rets, 5))
    cvar_95 = float(np.mean(rets[rets <= var_95]))

    return {
        "cagr_%":             round(cagr * 100, 2),
        "volatility_ann_%":   round(vol_ann * 100, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "sortino_ratio":      round(sortino, 3),
        "calmar_ratio":       round(calmar, 3),
        "max_drawdown_%":     round(max_dd * 100, 2),
        "max_dd_duration_days": dd_duration,
        "var_95_%":           round(var_95 * 100, 3),
        "cvar_95_%":          round(cvar_95 * 100, 3),
        "final_nav_$":        round(float(nav[-1]), 2),
        "total_return_%":     round((nav[-1] / nav[0] - 1) * 100, 2),
    }


# ---------------------------------------------------------------------------
# 5. P&L attribution by sentiment label
# ---------------------------------------------------------------------------

def pnl_attribution(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Break down realised P&L by sentiment label and signal direction.

    Parameters
    ----------
    trades_df : pd.DataFrame  Output of run_backtest().

    Returns
    -------
    pd.DataFrame  P&L summary by sentiment_label and signal.
    """
    if len(trades_df) == 0:
        return pd.DataFrame()

    attribution = (
        trades_df.groupby(["sentiment_label", "signal"])
        .agg(
            n_trades    = ("net_pnl", "count"),
            total_pnl   = ("net_pnl", "sum"),
            avg_pnl     = ("net_pnl", "mean"),
            win_rate    = ("net_pnl", lambda x: (x > 0).mean()),
            avg_score   = ("composite_score", "mean"),
        )
        .round(4)
    )
    attribution.index.names = ["sentiment", "signal"]
    return attribution


# ---------------------------------------------------------------------------
# 6. Pretty print summary
# ---------------------------------------------------------------------------

def print_performance_summary(
    backtest_stats: dict,
    ic_stats: dict,
    car_table: pd.DataFrame,
    quintile_table: pd.DataFrame,
) -> None:
    """
    Print a clean, ASCII-safe performance summary to stdout.
    """
    sep = "=" * 60

    print(sep)
    print("  FINBERT EARNINGS ALPHA - PERFORMANCE SUMMARY")
    print(sep)

    print("\n--- Portfolio Metrics ---")
    for k, v in backtest_stats.items():
        print(f"  {k:<35} {v}")

    print("\n--- Information Coefficient ---")
    for k, v in ic_stats.items():
        print(f"  {k:<35} {v}")

    print("\n--- CAR by Sentiment Bucket (CAR -1 to +5) ---")
    print(car_table.to_string())

    print("\n--- Quintile Forward Return Analysis ---")
    print(quintile_table.to_string())

    print(sep)
