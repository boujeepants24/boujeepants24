"""
plots.py
--------
Six publication-quality charts saved to results/.

1. Equity curve  : Strategy vs SPY vs 60/40 (hypothetical)
2. Drawdown      : Drawdown over time for strategy vs SPY
3. Monthly heatmap  : Calendar heatmap of monthly returns
4. Long vs Short attribution  : Cumulative return of each leg
5. Rolling Sharpe  : 252-day rolling Sharpe ratio
6. Kalman beta over time  : Average portfolio beta evolution

All text is ASCII-safe (no unicode characters).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all platforms)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

from .metrics import drawdown_series, monthly_returns, rolling_sharpe


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.dpi"     : 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size"      : 10,
    "axes.titlesize" : 12,
    "axes.labelsize" : 10,
})

STRATEGY_COLOR  = "#2196F3"   # blue
SPY_COLOR       = "#FF5722"   # red-orange
SIXFORTY_COLOR  = "#9C27B0"   # purple
LONG_COLOR      = "#4CAF50"   # green
SHORT_COLOR     = "#F44336"   # red


# ------------------------------------------------------------------ #
#  1. Equity Curve                                                    #
# ------------------------------------------------------------------ #

def plot_equity_curve(
    equity_curve: pd.Series,
    spy_equity: pd.Series,
    out_dir: str = RESULTS_DIR,
) -> str:
    """Plot strategy, SPY, and a hypothetical 60/40 equity curve."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 60/40 = 0.6 * SPY + 0.4 * bonds proxy (bonds return ~0% in this era)
    # Approximate bonds by flat 2% annual return
    n_days = len(spy_equity)
    bond_daily = np.exp(np.log(1.02) / 252)
    bond_series = pd.Series(
        bond_daily ** np.arange(n_days),
        index=spy_equity.index,
    )
    sixforty = 0.6 * spy_equity + 0.4 * bond_series
    sixforty = sixforty / sixforty.iloc[0]

    ax.plot(equity_curve.index, equity_curve.values,
            color=STRATEGY_COLOR, linewidth=2.0, label="Strategy (Kalman Factor-Neutral)", zorder=3)
    ax.plot(spy_equity.index,   spy_equity.values,
            color=SPY_COLOR,      linewidth=1.5, label="SPY (Buy & Hold)",  zorder=2)
    ax.plot(sixforty.index,     sixforty.values,
            color=SIXFORTY_COLOR, linewidth=1.5, label="60/40 (Approx)",    zorder=2, linestyle="--")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title("Cumulative Performance: Strategy vs Benchmarks")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(out_dir, "01_equity_curve.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
#  2. Drawdown                                                        #
# ------------------------------------------------------------------ #

def plot_drawdown(
    equity_curve: pd.Series,
    spy_equity: pd.Series,
    out_dir: str = RESULTS_DIR,
) -> str:
    """Plot drawdown series for strategy and SPY."""
    dd_strategy = drawdown_series(equity_curve) * 100
    dd_spy      = drawdown_series(spy_equity)   * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(dd_strategy.index, dd_strategy.values, 0,
                    color=STRATEGY_COLOR, alpha=0.6, label="Strategy")
    ax.fill_between(dd_spy.index, dd_spy.values, 0,
                    color=SPY_COLOR, alpha=0.3, label="SPY")

    ax.set_title("Drawdown Analysis")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(out_dir, "02_drawdown.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
#  3. Monthly Returns Heatmap                                         #
# ------------------------------------------------------------------ #

def plot_monthly_heatmap(
    daily_returns: pd.Series,
    out_dir: str = RESULTS_DIR,
) -> str:
    """Calendar heatmap of monthly strategy returns."""
    heatmap = monthly_returns(daily_returns)

    fig, ax = plt.subplots(figsize=(14, max(4, len(heatmap) * 0.55)))

    # Build annotation matrix with formatted strings
    # pandas >= 2.1 renamed DataFrame.applymap -> DataFrame.map
    import pandas as _pd
    _pd_major = int(_pd.__version__.split(".")[0])
    _pd_minor = int(_pd.__version__.split(".")[1])
    if (_pd_major, _pd_minor) >= (2, 1):
        annot = heatmap.map(lambda v: f"{v:.1f}" if not np.isnan(v) else "")
    else:
        annot = heatmap.applymap(lambda v: f"{v:.1f}" if not np.isnan(v) else "")

    sns.heatmap(
        heatmap,
        annot=annot,
        fmt="",
        cmap=sns.diverging_palette(10, 133, as_cmap=True),
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Return (%)"},
        annot_kws={"size": 8},
    )

    ax.set_title("Monthly Returns Heatmap (%)")
    ax.set_xlabel("")
    ax.set_ylabel("Year")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    path = os.path.join(out_dir, "03_monthly_heatmap.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
#  4. Long vs Short Leg Attribution                                   #
# ------------------------------------------------------------------ #

def plot_leg_attribution(
    long_returns: pd.Series,
    short_returns: pd.Series,
    out_dir: str = RESULTS_DIR,
) -> str:
    """Plot cumulative returns for the long and short legs separately."""
    long_equity  = np.exp(long_returns.cumsum())
    long_equity  = long_equity / long_equity.iloc[0]
    short_equity = np.exp(short_returns.cumsum())
    short_equity = short_equity / short_equity.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(long_equity.index,  long_equity.values,
            color=LONG_COLOR,  linewidth=2.0, label="Long Leg  (top 8)")
    ax.plot(short_equity.index, short_equity.values,
            color=SHORT_COLOR, linewidth=2.0, label="Short Leg (bottom 8)")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_title("Long vs Short Leg Attribution")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(out_dir, "04_leg_attribution.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
#  5. Rolling Sharpe Ratio                                            #
# ------------------------------------------------------------------ #

def plot_rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 252,
    out_dir: str = RESULTS_DIR,
) -> str:
    """Plot rolling 252-day annualised Sharpe ratio."""
    rs = rolling_sharpe(daily_returns, window=window)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(rs.index, rs.values, color=STRATEGY_COLOR, linewidth=1.5)
    ax.axhline(0.0, color="black",  linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(1.0, color=LONG_COLOR, linewidth=0.8, linestyle="--", alpha=0.5,
               label="Sharpe = 1.0")
    ax.fill_between(rs.index, rs.values, 0,
                    where=rs.values > 0, alpha=0.15, color=LONG_COLOR)
    ax.fill_between(rs.index, rs.values, 0,
                    where=rs.values < 0, alpha=0.15, color=SHORT_COLOR)

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio (annualised)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(out_dir, "05_rolling_sharpe.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
#  6. Kalman Beta Over Time                                           #
# ------------------------------------------------------------------ #

def plot_kalman_betas(
    beta_df: pd.DataFrame,
    weights_history: dict,
    out_dir: str = RESULTS_DIR,
) -> str:
    """
    Plot the average absolute Kalman beta of the active portfolio positions
    over time (i.e. the 'net beta before hedging' on each trading day).
    """
    sorted_dates = sorted(weights_history.keys())

    def active_tickers_on(date):
        active = [d for d in sorted_dates if d <= date]
        if not active:
            return []
        w = weights_history[active[-1]]
        return [k for k in w if k != "SPY"]

    dates  = []
    avg_betas = []
    net_betas = []

    for dt in beta_df.index:
        tickers = active_tickers_on(dt)
        if not tickers:
            continue
        tickers_avail = [t for t in tickers if t in beta_df.columns]
        if not tickers_avail:
            continue

        betas_now = beta_df.loc[dt, tickers_avail]
        dates.append(dt)
        avg_betas.append(betas_now.abs().mean())

        # Net portfolio beta (weighted)
        active = [d for d in sorted_dates if d <= dt]
        if active:
            w = weights_history[active[-1]]
            net = sum(w.get(t, 0.0) * beta_df.loc[dt, t]
                      for t in tickers_avail if t in beta_df.columns)
            net_betas.append(net)
        else:
            net_betas.append(np.nan)

    dates     = pd.DatetimeIndex(dates)
    avg_series = pd.Series(avg_betas, index=dates)
    net_series = pd.Series(net_betas, index=dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(avg_series.index, avg_series.values,
             color=STRATEGY_COLOR, linewidth=1.2)
    ax1.set_title("Average Absolute Kalman Beta of Active Positions")
    ax1.set_ylabel("Avg |Beta|")
    ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)

    ax2.plot(net_series.index, net_series.values,
             color=SPY_COLOR, linewidth=1.2, label="Net beta (before SPY hedge)")
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.set_title("Net Portfolio Beta (Pre-Hedge) over Time")
    ax2.set_ylabel("Net Beta")
    ax2.set_xlabel("Date")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(out_dir, "06_kalman_betas.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
#  Master function: save all plots                                    #
# ------------------------------------------------------------------ #

def save_all_plots(backtest_results: dict, beta_df: pd.DataFrame) -> list[str]:
    """
    Generate and save all six charts. Returns list of file paths.
    """
    print("Saving charts to results/ ...")
    paths = []

    paths.append(plot_equity_curve(
        backtest_results["equity_curve"],
        backtest_results["spy_equity"],
    ))
    paths.append(plot_drawdown(
        backtest_results["equity_curve"],
        backtest_results["spy_equity"],
    ))
    paths.append(plot_monthly_heatmap(
        backtest_results["daily_returns"],
    ))
    paths.append(plot_leg_attribution(
        backtest_results["long_returns"],
        backtest_results["short_returns"],
    ))
    paths.append(plot_rolling_sharpe(
        backtest_results["daily_returns"],
    ))
    paths.append(plot_kalman_betas(
        beta_df,
        backtest_results["weights_history"],
    ))

    for p in paths:
        print(f"  Saved: {os.path.basename(p)}")
    return paths
