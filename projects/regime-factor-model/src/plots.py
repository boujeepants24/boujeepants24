"""
plots.py
--------
All visualization logic for the Regime-Aware Multi-Factor Model.

Charts saved to results/ directory:
  1. equity_curves.png          -- Strategy vs SPY cumulative return
  2. drawdowns.png               -- Drawdown comparison
  3. regime_overlay.png          -- Price chart with regime backgrounds
  4. factor_attribution.png      -- Avg factor z-score by regime
  5. oos_vs_is.png               -- IS vs OOS performance bars
  6. regime_distribution.png     -- Pie chart of time spent in each regime
  7. monthly_returns_heatmap.png -- Monthly return heatmap
  8. rolling_sharpe.png          -- 6-month rolling Sharpe comparison
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates  as mdates
import seaborn           as sns

from src.metrics import (
    cumulative_return_series,
    drawdown_series,
    sharpe_ratio,
)
from src.regime import REGIME_NAMES, REGIME_BULL, REGIME_BEAR, REGIME_HIGH_VOL
from src.factors import FACTOR_NAMES
from src.backtest import BacktestResult

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

PALETTE = {
    "strategy":  "#2196F3",   # blue
    "benchmark": "#FF5722",   # deep orange
    "bull":      "#4CAF50",   # green
    "bear":      "#F44336",   # red
    "high_vol":  "#FF9800",   # amber
    "neutral":   "#9E9E9E",   # grey
}

REGIME_COLORS = {
    REGIME_BULL:     PALETTE["bull"],
    REGIME_BEAR:     PALETTE["bear"],
    REGIME_HIGH_VOL: PALETTE["high_vol"],
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to results/ directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def _apply_style() -> None:
    """Apply a clean, dark-themed matplotlib style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.labelcolor":  "#e0e0e0",
        "axes.edgecolor":   "#444466",
        "text.color":       "#e0e0e0",
        "xtick.color":      "#b0b0c0",
        "ytick.color":      "#b0b0c0",
        "grid.color":       "#2a2a4a",
        "grid.alpha":       0.5,
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444466",
        "legend.labelcolor": "#e0e0e0",
    })


# ---------------------------------------------------------------------------
# 1. Equity curves
# ---------------------------------------------------------------------------

def plot_equity_curves(result: BacktestResult) -> None:
    """Strategy vs SPY cumulative return chart with regime background shading."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    strat_cum = cumulative_return_series(result.portfolio_returns)
    bench_cum = cumulative_return_series(result.benchmark_returns)

    # Regime background shading
    if result.regime_series is not None:
        _shade_regimes(ax, result.regime_series, strat_cum.index)

    ax.plot(strat_cum.index, strat_cum.values, color=PALETTE["strategy"],
            linewidth=2.0, label="Regime-Factor Strategy", zorder=3)
    ax.plot(bench_cum.index, bench_cum.values, color=PALETTE["benchmark"],
            linewidth=1.5, linestyle="--", label="SPY Benchmark", zorder=3, alpha=0.85)

    ax.set_title("Regime-Aware Multi-Factor Model -- Cumulative Return (Growth of $1)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Legend with regime patches
    regime_patches = [
        mpatches.Patch(color=PALETTE["bull"],     alpha=0.25, label="Bull Regime"),
        mpatches.Patch(color=PALETTE["bear"],     alpha=0.25, label="Bear Regime"),
        mpatches.Patch(color=PALETTE["high_vol"], alpha=0.25, label="High-Vol Regime"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + regime_patches, loc="upper left", fontsize=9)

    fig.tight_layout()
    _save(fig, "equity_curves.png")


# ---------------------------------------------------------------------------
# 2. Drawdowns
# ---------------------------------------------------------------------------

def plot_drawdowns(result: BacktestResult) -> None:
    """Underwater equity chart comparing strategy and benchmark drawdowns."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    strat_dd = drawdown_series(result.portfolio_returns) * 100
    bench_dd = drawdown_series(result.benchmark_returns) * 100

    ax.fill_between(strat_dd.index, strat_dd.values, 0,
                    color=PALETTE["strategy"], alpha=0.6, label="Strategy Drawdown")
    ax.fill_between(bench_dd.index, bench_dd.values, 0,
                    color=PALETTE["benchmark"], alpha=0.35, label="SPY Drawdown")

    ax.set_title("Drawdown Analysis -- Strategy vs SPY")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left")

    fig.tight_layout()
    _save(fig, "drawdowns.png")


# ---------------------------------------------------------------------------
# 3. Regime overlay (SPY price + regime shading)
# ---------------------------------------------------------------------------

def plot_regime_overlay(result: BacktestResult, bench_close: pd.Series) -> None:
    """
    SPY price chart with colored background bands for each detected regime.
    Also shows transition points as vertical lines.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    if result.regime_series is not None:
        _shade_regimes(ax, result.regime_series, bench_close.index)

    bench_aligned = bench_close.reindex(
        bench_close.index.intersection(result.regime_series.index
                                       if result.regime_series is not None
                                       else bench_close.index)
    )
    ax.plot(bench_aligned.index, bench_aligned.values,
            color="white", linewidth=1.5, label="SPY Price", zorder=3)

    ax.set_title("HMM Regime Detection -- SPY Price with Regime Background")
    ax.set_ylabel("SPY Price ($)")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    patches = [
        mpatches.Patch(color=PALETTE["bull"],     alpha=0.4, label="Bull"),
        mpatches.Patch(color=PALETTE["bear"],     alpha=0.4, label="Bear"),
        mpatches.Patch(color=PALETTE["high_vol"], alpha=0.4, label="High-Vol"),
    ]
    ax.legend(handles=patches, loc="upper left")
    fig.tight_layout()
    _save(fig, "regime_overlay.png")


def _shade_regimes(ax: plt.Axes, regime_series: pd.Series,
                   date_index: pd.Index) -> None:
    """
    Draw colored vertical spans on ax corresponding to each regime period.
    Works by finding contiguous blocks of the same regime label.
    """
    regime_aligned = regime_series.reindex(date_index, method="ffill").dropna()
    if regime_aligned.empty:
        return

    # Find contiguous blocks
    changes = regime_aligned[regime_aligned != regime_aligned.shift(1)]
    starts  = changes.index.tolist()
    ends    = starts[1:] + [date_index[-1]]

    for start, end, reg in zip(starts, ends, changes.values):
        color = REGIME_COLORS.get(int(reg), PALETTE["neutral"])
        ax.axvspan(start, end, alpha=0.18, color=color, linewidth=0)


# ---------------------------------------------------------------------------
# 4. Factor attribution
# ---------------------------------------------------------------------------

def plot_factor_attribution(result: BacktestResult) -> None:
    """
    Bar chart showing average factor z-score of selected portfolios,
    grouped by regime.
    """
    if result.factor_attribution is None or result.factor_attribution.empty:
        print("  [WARN] No factor attribution data -- skipping plot")
        return

    _apply_style()

    # Merge regime into attribution df
    attr = result.factor_attribution.copy()
    if result.regime_series is not None:
        attr["regime"] = result.regime_series.reindex(attr.index, method="ffill")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: overall average factor contribution ---
    overall_mean = attr[FACTOR_NAMES].mean()
    colors_bar   = [PALETTE["strategy"]] * 4
    axes[0].bar(FACTOR_NAMES, overall_mean.values, color=colors_bar, alpha=0.85,
                edgecolor="#444466")
    axes[0].axhline(0, color="white", linewidth=0.8, linestyle="--")
    axes[0].set_title("Overall Avg Factor Z-Score (Selected Portfolio)")
    axes[0].set_ylabel("Average Z-Score")
    axes[0].set_xticklabels(["Momentum", "Quality", "Low-Vol", "Value"],
                             rotation=15)

    # --- Right: by regime ---
    if "regime" in attr.columns:
        regime_avg = attr.groupby("regime")[FACTOR_NAMES].mean()
        x     = np.arange(len(FACTOR_NAMES))
        width = 0.25
        for i, (regime_id, row) in enumerate(regime_avg.iterrows()):
            color = REGIME_COLORS.get(int(regime_id), PALETTE["neutral"])
            label = REGIME_NAMES.get(int(regime_id), f"Regime {regime_id}")
            axes[1].bar(x + i * width, row.values, width,
                        label=label, color=color, alpha=0.85, edgecolor="#444466")

        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(["Momentum", "Quality", "Low-Vol", "Value"],
                                 rotation=15)
        axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--")
        axes[1].set_title("Avg Factor Z-Score by Detected Regime")
        axes[1].set_ylabel("Average Z-Score")
        axes[1].legend(fontsize=9)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    _save(fig, "factor_attribution.png")


# ---------------------------------------------------------------------------
# 5. OOS vs IS performance bars
# ---------------------------------------------------------------------------

def plot_oos_vs_is(result: BacktestResult, metrics_df: pd.DataFrame) -> None:
    """
    Grouped bar chart showing IS vs OOS metrics side by side.
    Also shows per-window OOS returns as a separate bar chart.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: OOS window returns ---
    if result.oos_windows:
        wins   = [f"W{w['window']}" for w in result.oos_windows]
        rets   = [w["oos_return"] * 100 for w in result.oos_windows]
        colors = [PALETTE["bull"] if r > 0 else PALETTE["bear"] for r in rets]

        axes[0].bar(wins, rets, color=colors, alpha=0.85, edgecolor="#444466")
        axes[0].axhline(0, color="white", linewidth=0.8, linestyle="--")
        axes[0].set_title("OOS Window Returns (Walk-Forward)")
        axes[0].set_ylabel("6-Month OOS Return (%)")
        axes[0].set_xlabel("Walk-Forward Window")
    else:
        axes[0].text(0.5, 0.5, "No OOS windows", transform=axes[0].transAxes,
                     ha="center", va="center", color="white")

    # --- Right: IS vs OOS key metrics ---
    metric_labels = ["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    if result.is_returns is not None and len(result.is_returns) > 10:
        from src.metrics import cagr, sharpe_ratio, max_drawdown
        is_vals = [
            cagr(result.is_returns) * 100,
            sharpe_ratio(result.is_returns),
            max_drawdown(result.is_returns) * 100,
        ]
        oos_vals = [
            cagr(result.oos_returns) * 100 if result.oos_returns is not None else 0,
            sharpe_ratio(result.oos_returns) if result.oos_returns is not None else 0,
            max_drawdown(result.oos_returns) * 100 if result.oos_returns is not None else 0,
        ]

        x = np.arange(len(metric_labels))
        w = 0.35
        axes[1].bar(x - w/2, is_vals,  w, label="In-Sample",     color=PALETTE["strategy"],  alpha=0.85)
        axes[1].bar(x + w/2, oos_vals, w, label="Out-of-Sample",  color=PALETTE["benchmark"], alpha=0.85)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metric_labels, rotation=10)
        axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--")
        axes[1].set_title("In-Sample vs Out-of-Sample Performance")
        axes[1].legend()
    else:
        # Fallback: just show overall strategy vs SPY metrics
        if not metrics_df.empty:
            strat_sharpe = metrics_df.loc["Sharpe Ratio", metrics_df.columns[0]] \
                if "Sharpe Ratio" in metrics_df.index else 0
            bench_sharpe = metrics_df.loc["Sharpe Ratio", "SPY Benchmark"] \
                if "Sharpe Ratio" in metrics_df.index else 0
            axes[1].bar(["Strategy", "SPY"],
                        [float(strat_sharpe), float(bench_sharpe)],
                        color=[PALETTE["strategy"], PALETTE["benchmark"]],
                        alpha=0.85)
            axes[1].set_title("Sharpe Ratio: Strategy vs SPY")
            axes[1].set_ylabel("Sharpe Ratio")

    fig.tight_layout()
    _save(fig, "oos_vs_is.png")


# ---------------------------------------------------------------------------
# 6. Regime distribution pie
# ---------------------------------------------------------------------------

def plot_regime_distribution(result: BacktestResult) -> None:
    """Pie chart of fraction of trading days spent in each regime."""
    if result.regime_series is None or result.regime_series.empty:
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    counts = result.regime_series.value_counts().sort_index()
    labels = [REGIME_NAMES.get(int(k), f"State {k}") for k in counts.index]
    colors = [REGIME_COLORS.get(int(k), PALETTE["neutral"]) for k in counts.index]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
    )
    for t in texts + autotexts:
        t.set_color("#e0e0e0")

    ax.set_title("Time Spent in Each Market Regime (%)")
    fig.tight_layout()
    _save(fig, "regime_distribution.png")


# ---------------------------------------------------------------------------
# 7. Monthly returns heatmap
# ---------------------------------------------------------------------------

def plot_monthly_returns_heatmap(result: BacktestResult) -> None:
    """
    Calendar heatmap of monthly returns (year x month grid).
    """
    _apply_style()

    monthly = result.portfolio_returns.resample("ME").sum() * 100
    monthly_df = monthly.to_frame("ret")
    monthly_df["year"]  = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.65)))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="#1a1a2e",
        cbar_kws={"label": "Monthly Return (%)"},
    )
    ax.set_title("Monthly Returns Heatmap (%)")
    ax.set_ylabel("Year")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, "monthly_returns_heatmap.png")


# ---------------------------------------------------------------------------
# 8. Rolling Sharpe
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(result: BacktestResult, window: int = 126) -> None:
    """
    6-month (126-day) rolling Sharpe ratio for strategy and benchmark.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    def rolling_sharpe(rets: pd.Series, w: int) -> pd.Series:
        mean_ = rets.rolling(w).mean()
        std_  = rets.rolling(w).std(ddof=1)
        return (mean_ / std_.replace(0, np.nan)) * np.sqrt(252)

    strat_rs = rolling_sharpe(result.portfolio_returns, window)
    bench_rs = rolling_sharpe(result.benchmark_returns, window)

    ax.plot(strat_rs.index, strat_rs.values,
            color=PALETTE["strategy"], linewidth=1.8, label="Strategy")
    ax.plot(bench_rs.index, bench_rs.values,
            color=PALETTE["benchmark"], linewidth=1.4, linestyle="--",
            label="SPY", alpha=0.8)
    ax.axhline(0, color="white", linewidth=0.8, linestyle=":")
    ax.axhline(1, color="#4CAF50", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio -- Strategy vs SPY")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()

    fig.tight_layout()
    _save(fig, "rolling_sharpe.png")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(result: BacktestResult, bench_close: pd.Series,
                       metrics_df: pd.DataFrame) -> None:
    """Generate and save all charts."""
    print("\nGenerating charts ...")

    plot_equity_curves(result)
    plot_drawdowns(result)
    plot_regime_overlay(result, bench_close)
    plot_factor_attribution(result)
    plot_oos_vs_is(result, metrics_df)
    plot_regime_distribution(result)
    plot_monthly_returns_heatmap(result)
    plot_rolling_sharpe(result)

    print(f"  All charts saved to: {RESULTS_DIR}")
