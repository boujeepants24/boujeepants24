"""
plots.py
--------
All visualization for the funding rate arbitrage strategy.

Charts produced
---------------
1. Equity curve vs BTC buy-and-hold
2. Funding rate heatmap across assets over time
3. P&L attribution stacked bar chart (monthly)
4. Distribution of 8h returns (histogram + KDE)
5. Drawdown analysis
6. Funding rate autocorrelation (ACF) by asset
7. Annualized funding rate time series (per asset)
8. Basis spread time series (perp vs spot premium)
9. Monthly funding yield heatmap
10. Leverage and position count over time

All plots are saved to an output directory; no interactive display is required.
Style: clean, professional (seaborn whitegrid), high DPI.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Optional

from metrics import (
    drawdown_series,
    monthly_pnl_attribution,
    compute_period_returns,
    PERIODS_PER_YEAR,
)
from signals import compute_autocorrelation, annualize_funding, compute_basis

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        120,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

COLORS = {
    "funding":  "#2ecc71",
    "basis":    "#e74c3c",
    "fees":     "#e67e22",
    "borrow":   "#9b59b6",
    "net":      "#3498db",
    "equity":   "#2c3e50",
    "btc":      "#f39c12",
    "drawdown": "#c0392b",
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    """Save figure and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _format_dates(ax, freq: str = "6ME") -> None:
    """Apply consistent date formatting to an x-axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ---------------------------------------------------------------------------
# 1. Equity Curve vs BTC Buy-and-Hold
# ---------------------------------------------------------------------------

def plot_equity_curve(
    ledger:       pd.DataFrame,
    btc_ohlcv:    pd.DataFrame,
    initial_nav:  float,
    output_dir:   str,
) -> None:
    """
    Overlay the strategy NAV against a BTC buy-and-hold benchmark,
    both normalized to 1.0 at the start.
    """
    nav    = ledger["nav"]
    strat  = nav / initial_nav

    # BTC normalized: align to strategy start
    btc_close = btc_ohlcv["close"].reindex(nav.index, method="ffill").dropna()
    btc_norm  = btc_close / btc_close.iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.suptitle("Strategy Equity Curve vs BTC Buy-and-Hold", fontsize=14, fontweight="bold")

    # Equity curves
    ax1.plot(strat.index, strat.values,   label="Funding Arb Strategy",
             color=COLORS["equity"], linewidth=2.0)
    ax1.plot(btc_norm.index, btc_norm.values, label="BTC Buy-and-Hold",
             color=COLORS["btc"], linewidth=1.5, alpha=0.8, linestyle="--")
    ax1.set_ylabel("Normalized NAV (start = 1.0)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}x"))

    # Drawdown
    dd = drawdown_series(nav)
    ax2.fill_between(dd.index, dd.values * 100, 0,
                     color=COLORS["drawdown"], alpha=0.7, label="Strategy Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(top=0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    _format_dates(ax2)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "01_equity_curve.png"))


# ---------------------------------------------------------------------------
# 2. Funding Rate Heatmap
# ---------------------------------------------------------------------------

def plot_funding_heatmap(
    funding_wide: pd.DataFrame,
    output_dir:   str,
    resample:     str = "W",   # aggregate to weekly for readability
) -> None:
    """
    Heatmap: assets (rows) x time (columns), values = annualized funding rate (%).
    """
    # Resample to weekly mean and annualize
    weekly   = funding_wide.resample(resample).mean()
    ann_wide = (weekly * PERIODS_PER_YEAR * 100).T   # assets as rows, time as columns

    # Keep most recent 104 weeks (2 years) to avoid overlong x-axis
    ann_wide = ann_wide.iloc[:, -104:]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("Annualized Funding Rate Heatmap (% per year)", fontsize=14, fontweight="bold")

    sns.heatmap(
        ann_wide,
        ax          = ax,
        cmap        = "RdYlGn",
        center      = 0,
        vmin        = -100,
        vmax        = 100,
        linewidths  = 0.3,
        cbar_kws    = {"label": "Annualized Rate (%)"},
        xticklabels = max(1, ann_wide.shape[1] // 20),
    )

    # Format x-tick labels as dates
    tick_labels = [str(c.date()) if hasattr(c, "date") else str(c) for c in ann_wide.columns]
    step = max(1, len(tick_labels) // 15)
    ax.set_xticks(range(0, len(tick_labels), step))
    ax.set_xticklabels(tick_labels[::step], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Asset")

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "02_funding_rate_heatmap.png"))


# ---------------------------------------------------------------------------
# 3. Monthly P&L Attribution (Stacked Bar)
# ---------------------------------------------------------------------------

def plot_monthly_attribution(
    ledger:     pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Stacked bar chart showing monthly P&L broken down by:
      funding income, basis P&L, fee cost, borrow cost
    """
    monthly = monthly_pnl_attribution(ledger)

    # Convert index to string labels
    labels = [str(p) for p in monthly.index]
    x      = np.arange(len(labels))

    bar_w  = 0.65
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4), 7))
    fig.suptitle("Monthly P&L Attribution (USD)", fontsize=14, fontweight="bold")

    bottom_pos = np.zeros(len(monthly))
    bottom_neg = np.zeros(len(monthly))

    components = [
        ("funding_income", "Funding Income",  COLORS["funding"]),
        ("basis_pnl",      "Basis P&L",       COLORS["basis"]),
    ]
    neg_components = [
        ("fee_cost",       "Fee Cost",        COLORS["fees"]),
        ("borrow_cost",    "Borrow Cost",      COLORS["borrow"]),
    ]

    # Positive components (stack upward)
    for col, label, color in components:
        vals = monthly[col].values
        pos  = np.maximum(vals, 0)
        neg  = np.minimum(vals, 0)
        ax.bar(x, pos, bar_w, bottom=bottom_pos, color=color, label=label, alpha=0.85)
        ax.bar(x, neg, bar_w, bottom=bottom_neg, color=color, alpha=0.85)
        bottom_pos += pos
        bottom_neg += neg

    # Negative components (stack downward)
    for col, label, color in neg_components:
        vals = -monthly[col].values   # make costs negative
        ax.bar(x, vals, bar_w, bottom=bottom_neg, color=color, label=label, alpha=0.85)
        bottom_neg += vals

    # Net P&L as a line overlay
    ax2 = ax.twinx()
    ax2.plot(x, monthly["net_pnl"].values / 1000,
             color=COLORS["net"], linewidth=2.0, marker="o", markersize=4, label="Net P&L")
    ax2.set_ylabel("Net P&L ($K)", color=COLORS["net"])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}K"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("P&L (USD)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))

    # Combine legends
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "03_monthly_attribution.png"))


# ---------------------------------------------------------------------------
# 4. Return Distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(
    ledger:     pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Histogram + KDE of 8h returns, with annotations for skewness and kurtosis.
    A right-skewed distribution indicates consistent small gains + rare losses.
    """
    nav     = ledger["nav"]
    returns = compute_period_returns(nav) * 100   # as percent

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Distribution of 8h Returns", fontsize=14, fontweight="bold")

    ax.hist(returns, bins=80, density=True, alpha=0.5, color=COLORS["equity"],
            label="Empirical distribution")

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde   = gaussian_kde(returns.dropna())
    x_rng = np.linspace(returns.min(), returns.max(), 400)
    ax.plot(x_rng, kde(x_rng), color=COLORS["net"], linewidth=2.0, label="KDE")

    # Normal reference
    from scipy.stats import norm
    mu, sig = returns.mean(), returns.std()
    ax.plot(x_rng, norm.pdf(x_rng, mu, sig), color="gray",
            linewidth=1.5, linestyle="--", label="Normal reference")

    # Annotations
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())
    ax.text(0.97, 0.95,
            f"Mean  : {mu:+.4f}%\n"
            f"Std   : {sig:.4f}%\n"
            f"Skew  : {skew:+.3f}\n"
            f"Kurt  : {kurt:+.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("8h Return (%)")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "04_return_distribution.png"))


# ---------------------------------------------------------------------------
# 5. Drawdown Analysis
# ---------------------------------------------------------------------------

def plot_drawdown(
    ledger:     pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Underwater plot of the strategy drawdown over the full backtest period.
    """
    nav = ledger["nav"]
    dd  = drawdown_series(nav) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Strategy Drawdown Analysis", fontsize=14, fontweight="bold")

    ax.fill_between(dd.index, dd.values, 0, color=COLORS["drawdown"], alpha=0.75)
    ax.plot(dd.index, dd.values, color=COLORS["drawdown"], linewidth=0.8)

    # Annotate maximum drawdown
    min_val = dd.min()
    min_ts  = dd.idxmin()
    ax.annotate(
        f"Max DD: {min_val:.1f}%",
        xy=(min_ts, min_val),
        xytext=(min_ts, min_val - 3),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_ylabel("Drawdown (%)")
    ax.set_ylim(top=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    _format_dates(ax)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "05_drawdown.png"))


# ---------------------------------------------------------------------------
# 6. Funding Rate Autocorrelation
# ---------------------------------------------------------------------------

def plot_autocorrelation(
    funding_wide: pd.DataFrame,
    output_dir:   str,
    max_lags:     int = 10,
) -> None:
    """
    Bar chart of funding rate ACF at lags 1..max_lags for each asset.
    High lag-1 ACF confirms that funding rates are persistent (key strategy premise).
    """
    n_assets = len(funding_wide.columns)
    ncols    = 4
    nrows    = (n_assets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    fig.suptitle("Funding Rate Autocorrelation by Asset (lags 1..10)",
                 fontsize=14, fontweight="bold")
    axes_flat = axes.flatten() if n_assets > 1 else [axes]

    for i, sym in enumerate(funding_wide.columns):
        ax   = axes_flat[i]
        acf  = compute_autocorrelation(funding_wide[sym].dropna(), max_lags=max_lags)
        lags = list(acf.index)
        vals = acf.values

        colors = [COLORS["funding"] if v >= 0 else COLORS["basis"] for v in vals]
        ax.bar(lags, vals, color=colors, width=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        # 95% confidence band for white noise: +/- 1.96 / sqrt(n)
        n = len(funding_wide[sym].dropna())
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci,  color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(-ci, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(sym, fontsize=10)
        ax.set_xlabel("Lag (8h periods)")
        ax.set_ylabel("ACF")
        ax.set_ylim(-0.3, 0.6)
        ax.set_xticks(lags)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "06_acf.png"))


# ---------------------------------------------------------------------------
# 7. Annualized Funding Rate Time Series
# ---------------------------------------------------------------------------

def plot_funding_timeseries(
    funding_wide: pd.DataFrame,
    output_dir:   str,
) -> None:
    """
    Multi-panel time series of annualized funding rates for all assets.
    """
    ann = annualize_funding(funding_wide) * 100   # in percent
    symbols  = ann.columns.tolist()
    n_assets = len(symbols)
    ncols    = 2
    nrows    = (n_assets + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), sharex=True)
    fig.suptitle("Annualized Funding Rate (%) - All Assets", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    for i, sym in enumerate(symbols):
        ax = axes_flat[i]
        s  = ann[sym].resample("D").mean().dropna()
        ax.plot(s.index, s.values, linewidth=0.8, color=COLORS["funding"])
        ax.fill_between(s.index, s.values, 0,
                        where=(s.values >= 0), alpha=0.25, color=COLORS["funding"])
        ax.fill_between(s.index, s.values, 0,
                        where=(s.values < 0),  alpha=0.25, color=COLORS["basis"])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(sym, fontsize=10)
        ax.set_ylabel("Ann. Rate (%)")
        _format_dates(ax)

    for j in range(len(symbols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "07_funding_timeseries.png"))


# ---------------------------------------------------------------------------
# 8. Basis Spread Time Series
# ---------------------------------------------------------------------------

def plot_basis_timeseries(
    perp_dict:  dict,
    spot_dict:  dict,
    output_dir: str,
) -> None:
    """
    Plot the basis (perp premium over spot) for all assets over time.
    """
    symbols  = list(perp_dict.keys())
    n_assets = len(symbols)
    ncols    = 2
    nrows    = (n_assets + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), sharex=True)
    fig.suptitle("Basis Spread: (Perp - Spot) / Spot * 100 (%)",
                 fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    for i, sym in enumerate(symbols):
        ax = axes_flat[i]
        if sym not in spot_dict:
            ax.set_visible(False)
            continue
        perp_cl = perp_dict[sym]["close"]
        spot_cl = spot_dict[sym]["close"].reindex(perp_cl.index, method="ffill")
        basis_s = compute_basis(perp_cl, spot_cl).resample("D").mean().dropna()

        ax.plot(basis_s.index, basis_s.values, linewidth=0.8, color=COLORS["basis"])
        ax.fill_between(basis_s.index, basis_s.values, 0,
                        where=(basis_s.values >= 0), alpha=0.2, color=COLORS["basis"])
        ax.fill_between(basis_s.index, basis_s.values, 0,
                        where=(basis_s.values < 0),  alpha=0.2, color=COLORS["net"])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(sym, fontsize=10)
        ax.set_ylabel("Basis (%)")
        _format_dates(ax)

    for j in range(len(symbols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "08_basis_timeseries.png"))


# ---------------------------------------------------------------------------
# 9. Monthly Funding Yield Heatmap
# ---------------------------------------------------------------------------

def plot_monthly_funding_heatmap(
    funding_wide: pd.DataFrame,
    output_dir:   str,
) -> None:
    """
    Heatmap: assets (rows) x month (columns), values = average annualized funding yield (%).
    """
    ann     = (annualize_funding(funding_wide) * 100)
    monthly = ann.resample("ME").mean()
    monthly.index = monthly.index.to_period("M").astype(str)

    pivot = monthly.T   # assets as rows, months as columns

    fig, ax = plt.subplots(figsize=(max(12, len(monthly) * 0.5), 5))
    fig.suptitle("Monthly Average Annualized Funding Yield (%) per Asset",
                 fontsize=14, fontweight="bold")

    sns.heatmap(
        pivot,
        ax         = ax,
        cmap       = "RdYlGn",
        center     = 0,
        vmin       = -200,
        vmax       = 200,
        linewidths = 0.5,
        annot      = len(monthly) <= 30,   # only annotate if not too many months
        fmt        = ".0f",
        cbar_kws   = {"label": "Ann. Funding (%)"},
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Asset")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "09_monthly_funding_heatmap.png"))


# ---------------------------------------------------------------------------
# 10. Leverage and Position Count
# ---------------------------------------------------------------------------

def plot_leverage_and_positions(
    ledger:     pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Dual-axis chart: leverage (left) and number of open positions (right) over time.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.suptitle("Portfolio Leverage and Open Positions Over Time",
                 fontsize=14, fontweight="bold")

    lev = ledger["leverage"].resample("D").mean()
    ax1.fill_between(lev.index, lev.values, 0, alpha=0.4, color=COLORS["equity"])
    ax1.plot(lev.index, lev.values, linewidth=0.8, color=COLORS["equity"], label="Leverage")
    ax1.axhline(3.0, color="red", linewidth=1.0, linestyle="--", alpha=0.7, label="Max leverage")
    ax1.set_ylabel("Gross Leverage (x)")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    n_pos = ledger["n_positions"].resample("D").max()
    ax2.step(n_pos.index, n_pos.values, where="post",
             linewidth=1.2, color=COLORS["funding"], alpha=0.8, label="Open Positions")
    ax2.set_ylabel("Number of Open Positions")
    ax2.set_ylim(bottom=0)

    # Merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    _format_dates(ax1)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "10_leverage_positions.png"))


# ---------------------------------------------------------------------------
# Master: generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    ledger:       pd.DataFrame,
    funding_wide: pd.DataFrame,
    perp_dict:    dict,
    spot_dict:    dict,
    initial_nav:  float,
    output_dir:   str = "output/plots",
) -> None:
    """
    Generate and save all charts.

    Parameters
    ----------
    ledger       : per-period backtest ledger from backtest.py
    funding_wide : wide DataFrame of 8h funding rates
    perp_dict    : dict[symbol -> OHLCV DataFrame]
    spot_dict    : dict[symbol -> OHLCV DataFrame]
    initial_nav  : starting NAV
    output_dir   : directory path for saved PNG files
    """
    print(f"Generating plots -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    btc_perp = perp_dict.get("BTCUSDT", pd.DataFrame())

    plot_equity_curve(ledger, btc_perp, initial_nav, output_dir)
    plot_funding_heatmap(funding_wide, output_dir)
    plot_monthly_attribution(ledger, output_dir)
    plot_return_distribution(ledger, output_dir)
    plot_drawdown(ledger, output_dir)
    plot_autocorrelation(funding_wide, output_dir)
    plot_funding_timeseries(funding_wide, output_dir)
    plot_basis_timeseries(perp_dict, spot_dict, output_dir)
    plot_monthly_funding_heatmap(funding_wide, output_dir)
    plot_leverage_and_positions(ledger, output_dir)

    print("All plots generated.")
