"""
plots.py
--------
Visualization module for the FinBERT earnings alpha strategy.

Charts produced:
  1. Event Study Chart  — Average CAR(-5, +20) for positive/neutral/negative
                          sentiment groups, with confidence bands.
  2. Sentiment Scatter  — FinBERT composite score vs realized 5-day return.
  3. Quintile CAR Chart — Bar chart of mean CAR(-1,+5) by sentiment quintile.
  4. IC Time-Series     — Rolling monthly Information Coefficient.
  5. Equity Curve       — Strategy NAV vs SPY buy-and-hold (normalized).
  6. CAR Distribution   — Histogram of CAR(-1,+5) split by sentiment label.
  7. Market Model Fit   — Beta distribution across the universe.

All labels are ASCII-only to avoid Windows cp1252 encoding issues.
Charts are saved to the output/ directory as PNG files.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

matplotlib.use("Agg")   # Non-interactive backend (safe for scripts)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

PALETTE = {
    "positive": "#2ecc71",   # green
    "neutral":  "#3498db",   # blue
    "negative": "#e74c3c",   # red
    "strategy": "#2c3e50",   # dark
    "benchmark": "#95a5a6",  # grey
}

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "figure.dpi":       120,
    "savefig.bbox":     "tight",
})


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plots] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Event study chart
# ---------------------------------------------------------------------------

def plot_event_study(
    ar_panel: dict,
    events_scored: pd.DataFrame,
    pre_days: int = 5,
    post_days: int = 20,
) -> str:
    """
    Plot average CAR(-5, +20) for positive, neutral, and negative events.

    Parameters
    ----------
    ar_panel       : dict  (ticker, event_date) -> abnormal return DataFrame.
    events_scored  : pd.DataFrame  Must have ticker, earnings_date, sentiment_label.
    pre_days       : int   Days before event to show.
    post_days      : int   Days after event to show.

    Returns
    -------
    str  Path to saved PNG.
    """
    # Build a sentiment lookup
    sentiment_map = dict(
        zip(
            zip(events_scored["ticker"], pd.to_datetime(events_scored["earnings_date"])),
            events_scored["sentiment_label"],
        )
    )

    # Collect CAR time-series by sentiment
    car_by_sentiment: dict[str, list] = {"positive": [], "neutral": [], "negative": []}

    for (ticker, event_date), ar_df in ar_panel.items():
        key = (ticker, event_date)
        label = sentiment_map.get(key, None)
        if label not in car_by_sentiment:
            continue

        # Filter to event window
        sub = ar_df[
            (ar_df["event_day"] >= -pre_days) & (ar_df["event_day"] <= post_days)
        ].sort_values("event_day")

        if len(sub) < (pre_days + 5):  # require enough data
            continue

        # Cumulative abnormal return from day -pre_days
        sub = sub.copy()
        sub["car"] = sub["abnormal_return"].cumsum()
        car_by_sentiment[label].append(sub.set_index("event_day")["car"])

    event_days = range(-pre_days, post_days + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, series_list in car_by_sentiment.items():
        if not series_list:
            continue
        # Align all series to common event_day index
        aligned = pd.concat(series_list, axis=1).reindex(event_days)
        mean_car = aligned.mean(axis=1)
        se_car   = aligned.sem(axis=1)

        color = PALETTE[label]
        ax.plot(event_days, mean_car, color=color, linewidth=2, label=f"{label.capitalize()} (n={len(series_list)})")
        ax.fill_between(
            event_days,
            mean_car - 1.96 * se_car,
            mean_car + 1.96 * se_car,
            alpha=0.15,
            color=color,
        )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Announcement Day")
    ax.axhline(y=0, color="grey",  linestyle="-",  linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Trading Days Relative to Earnings Announcement")
    ax.set_ylabel("Cumulative Abnormal Return (CAR)")
    ax.set_title("Event Study: Average CAR by FinBERT Sentiment (-5 to +20 Days)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)

    return _save(fig, "01_event_study_car.png")


# ---------------------------------------------------------------------------
# 2. Sentiment score vs realized return scatter
# ---------------------------------------------------------------------------

def plot_sentiment_scatter(master_df: pd.DataFrame) -> str:
    """
    Scatter plot: FinBERT composite score vs 5-day forward return.

    Parameters
    ----------
    master_df : pd.DataFrame  Must have composite_score and forward_5d_return.

    Returns
    -------
    str  Path to saved PNG.
    """
    valid = master_df[["composite_score", "forward_5d_return", "sentiment_label"]].dropna()

    fig, ax = plt.subplots(figsize=(10, 7))

    for label in ["positive", "neutral", "negative"]:
        sub = valid[valid["sentiment_label"] == label]
        ax.scatter(
            sub["composite_score"],
            sub["forward_5d_return"],
            alpha=0.35,
            s=20,
            color=PALETTE[label],
            label=f"{label.capitalize()} (n={len(sub)})",
            rasterized=True,
        )

    # Regression line (all points)
    z = np.polyfit(valid["composite_score"], valid["forward_5d_return"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid["composite_score"].min(), valid["composite_score"].max(), 200)
    ax.plot(x_line, p(x_line), color="black", linewidth=2.0, label="OLS fit (all)")

    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("FinBERT Composite Sentiment Score (Positive - Negative Prob)")
    ax.set_ylabel("Realised 5-Day Forward Return (Log)")
    ax.set_title("Information Coefficient: FinBERT Score vs Realised Forward Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    return _save(fig, "02_ic_scatter.png")


# ---------------------------------------------------------------------------
# 3. Quintile CAR bar chart
# ---------------------------------------------------------------------------

def plot_quintile_cars(quintile_table: pd.DataFrame, car_col: str = "mean_car") -> str:
    """
    Bar chart of mean CAR(-1,+5) by sentiment quintile.

    Q1 = most negative sentiment, Q5 = most positive.
    A monotonically increasing bar chart validates the model.

    Parameters
    ----------
    quintile_table : pd.DataFrame  Output of signals.quintile_car_summary().
    car_col        : str           Column name for mean CAR.

    Returns
    -------
    str  Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    quintiles  = quintile_table.index.tolist()
    mean_cars  = quintile_table[car_col].values
    colors     = ["#e74c3c", "#e67e22", "#3498db", "#27ae60", "#2ecc71"]

    bars = ax.bar(quintiles, mean_cars, color=colors[:len(quintiles)], edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, mean_cars):
        y_pos = val + (0.0002 if val >= 0 else -0.0004)
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos,
                f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(quintiles)
    ax.set_xticklabels([f"Q{q}\n({'most neg' if q==1 else 'most pos' if q==5 else ''})" for q in quintiles])
    ax.set_xlabel("Sentiment Quintile (Q1=Most Negative, Q5=Most Positive)")
    ax.set_ylabel("Mean CAR(-1, +5)")
    ax.set_title("CAR by Sentiment Quintile — Monotonic Relationship Validates Model")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=2))
    ax.grid(True, axis="y", alpha=0.4)

    return _save(fig, "03_quintile_car.png")


# ---------------------------------------------------------------------------
# 4. IC time-series
# ---------------------------------------------------------------------------

def plot_ic_timeseries(ic_df: pd.DataFrame) -> str:
    """
    Plot monthly IC with rolling 3-month mean overlay.

    Parameters
    ----------
    ic_df : pd.DataFrame  Output of metrics.compute_ic_timeseries().

    Returns
    -------
    str  Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    ic_vals     = ic_df["ic"]
    roll_3m     = ic_vals.rolling(3).mean()

    ax.bar(ic_df.index, ic_vals, width=20, alpha=0.55, color="#3498db", label="Monthly IC")
    ax.plot(ic_df.index, roll_3m, color="#e74c3c", linewidth=2, label="3-Month Rolling IC")
    ax.axhline(y=0,    color="black", linewidth=0.8)
    ax.axhline(y=0.05, color="#2ecc71", linewidth=1.2, linestyle="--", alpha=0.7, label="IC=0.05 threshold")

    ax.set_xlabel("Date")
    ax.set_ylabel("Information Coefficient (Spearman)")
    ax.set_title("Monthly IC: FinBERT Composite Score vs 5-Day Forward Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    mean_ic = ic_vals.mean()
    ax.text(
        0.02, 0.95,
        f"Mean IC: {mean_ic:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return _save(fig, "04_ic_timeseries.png")


# ---------------------------------------------------------------------------
# 5. Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_df: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    initial_nav: float = 1_000_000.0,
) -> str:
    """
    Plot strategy equity curve vs SPY buy-and-hold (both normalized to 1.0).

    Parameters
    ----------
    equity_df        : pd.DataFrame  Output of run_backtest().
    prices           : pd.DataFrame  Price data including benchmark.
    benchmark_ticker : str           Column name for benchmark in prices.
    initial_nav      : float         Starting portfolio value.

    Returns
    -------
    str  Path to saved PNG.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Strategy NAV (normalized)
    strat_dates = pd.to_datetime(equity_df["date"])
    strat_nav   = equity_df["nav"] / initial_nav

    ax1.plot(strat_dates, strat_nav, color=PALETTE["strategy"], linewidth=1.8,
             label="FinBERT Earnings Alpha Strategy")

    # Benchmark (SPY normalized to 1.0 at strategy start)
    if benchmark_ticker in prices.columns:
        spy_prices = prices[benchmark_ticker].dropna()
        start_date = strat_dates.iloc[0]
        spy_sub    = spy_prices[spy_prices.index >= start_date]
        if len(spy_sub) > 0:
            spy_norm = spy_sub / spy_sub.iloc[0]
            ax1.plot(spy_sub.index, spy_norm, color=PALETTE["benchmark"], linewidth=1.4,
                     linestyle="--", label=f"{benchmark_ticker} Buy-and-Hold")

    ax1.axhline(y=1.0, color="black", linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("Normalized NAV (1.0 = Start)")
    ax1.set_title("FinBERT Earnings Alpha: Equity Curve vs Benchmark")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    # Drawdown subplot
    nav_arr  = equity_df["nav"].values
    cummax   = np.maximum.accumulate(nav_arr)
    drawdown = (nav_arr - cummax) / cummax

    ax2.fill_between(strat_dates, drawdown, 0, color="#e74c3c", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save(fig, "05_equity_curve.png")


# ---------------------------------------------------------------------------
# 6. CAR distribution by sentiment label
# ---------------------------------------------------------------------------

def plot_car_distribution(master_df: pd.DataFrame, car_col: str = "car_m1_p5") -> str:
    """
    Histogram of CAR(-1,+5) distributions for each sentiment label.

    Parameters
    ----------
    master_df : pd.DataFrame  Must have car_m1_p5 and sentiment_label.
    car_col   : str           CAR column name.

    Returns
    -------
    str  Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for label in ["positive", "neutral", "negative"]:
        subset = master_df.loc[master_df["sentiment_label"] == label, car_col].dropna()
        if len(subset) == 0:
            continue
        ax.hist(
            subset,
            bins=40,
            alpha=0.55,
            color=PALETTE[label],
            label=f"{label.capitalize()} (mean={subset.mean()*100:.2f}%, n={len(subset)})",
            density=True,
            edgecolor="white",
            linewidth=0.4,
        )
        # Vertical mean line
        ax.axvline(subset.mean(), color=PALETTE[label], linewidth=2, linestyle="--")

    ax.axvline(0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("CAR(-1, +5)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of CAR(-1, +5) by FinBERT Sentiment Label")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save(fig, "06_car_distribution.png")


# ---------------------------------------------------------------------------
# 7. Market model beta distribution
# ---------------------------------------------------------------------------

def plot_beta_distribution(event_study_results: pd.DataFrame) -> str:
    """
    Histogram of estimated market betas across the universe.

    Parameters
    ----------
    event_study_results : pd.DataFrame  Output of run_event_study().

    Returns
    -------
    str  Path to saved PNG.
    """
    valid_betas = event_study_results.loc[event_study_results["valid"], "beta"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(valid_betas, bins=30, color="#3498db", edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axvline(valid_betas.mean(), color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Mean beta = {valid_betas.mean():.3f}")
    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="-", alpha=0.6, label="Beta = 1.0")

    ax.set_xlabel("Estimated Market Beta (OLS, 120-day window)")
    ax.set_ylabel("Frequency")
    ax.set_title("Market Model Beta Distribution Across Events")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save(fig, "07_beta_distribution.png")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(
    ar_panel: dict,
    events_scored: pd.DataFrame,
    master_df: pd.DataFrame,
    quintile_table: pd.DataFrame,
    ic_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    prices: pd.DataFrame,
    event_study_results: pd.DataFrame,
) -> list[str]:
    """
    Generate all charts and return list of output file paths.

    Parameters
    ----------
    See individual plot functions for parameter descriptions.

    Returns
    -------
    list[str]  Paths to all saved PNG files.
    """
    paths = []

    print("[plots] Generating event study chart ...")
    paths.append(plot_event_study(ar_panel, events_scored))

    print("[plots] Generating IC scatter ...")
    paths.append(plot_sentiment_scatter(master_df))

    print("[plots] Generating quintile CAR chart ...")
    if len(quintile_table) > 0:
        paths.append(plot_quintile_cars(quintile_table))

    print("[plots] Generating IC time-series ...")
    if len(ic_df) > 0:
        paths.append(plot_ic_timeseries(ic_df))

    print("[plots] Generating equity curve ...")
    if len(equity_df) > 0:
        paths.append(plot_equity_curve(equity_df, prices))

    print("[plots] Generating CAR distribution ...")
    paths.append(plot_car_distribution(master_df))

    print("[plots] Generating beta distribution ...")
    paths.append(plot_beta_distribution(event_study_results))

    return paths
