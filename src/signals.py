"""
signals.py
----------
Signal construction from FinBERT sentiment scores.

Converts raw FinBERT composite scores into ranked trading signals
suitable for the event-driven backtester.

Signal logic:
  - At each earnings event, rank all stocks reporting within a 2-day window
    by composite_score (positive - negative probability).
  - Long the top-5 ranked stocks (most positive sentiment).
  - Short the bottom-5 ranked stocks (most negative sentiment).
  - Hold for 5 trading days post-announcement.
  - Positions are sized equally within long and short books.

The signal is constructed event-by-event rather than on a fixed schedule,
since earnings dates are staggered across the universe.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LONG   = 5   # Top N stocks to go long
N_SHORT  = 5   # Bottom N stocks to go short
HOLD_DAYS = 5  # Trading days to hold position


# ---------------------------------------------------------------------------
# Sentiment quintile ranking
# ---------------------------------------------------------------------------

def assign_quintile(series: pd.Series) -> pd.Series:
    """
    Assign sentiment quintile (1=most negative, 5=most positive) to scores.

    Parameters
    ----------
    series : pd.Series  Composite sentiment scores.

    Returns
    -------
    pd.Series  Integer quintile labels 1-5.
    """
    return pd.qcut(series.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)


# ---------------------------------------------------------------------------
# Signal generation per event batch
# ---------------------------------------------------------------------------

def generate_signals(
    events_scored: pd.DataFrame,
    n_long: int = N_LONG,
    n_short: int = N_SHORT,
    hold_days: int = HOLD_DAYS,
    group_window_days: int = 2,
) -> pd.DataFrame:
    """
    Generate long/short signals from scored earnings events.

    Events are grouped by announcement date (within a `group_window_days`
    window). Within each group, we rank by composite_score and assign
    LONG to top n_long and SHORT to bottom n_short.

    Parameters
    ----------
    events_scored    : pd.DataFrame
        Must have: ticker, earnings_date, composite_score.
    n_long           : int  Number of long positions per event batch.
    n_short          : int  Number of short positions per event batch.
    hold_days        : int  Trading days to hold each position.
    group_window_days: int  Days within which events are grouped together.

    Returns
    -------
    pd.DataFrame with columns:
        ticker, earnings_date, exit_day_offset, composite_score,
        signal (1=long, -1=short, 0=none), quintile, position_size
    """
    df = events_scored.copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values("earnings_date").reset_index(drop=True)

    # Assign each event to a batch by rounding date to nearest group_window_days
    # (simple approach: use floor division on day-of-year)
    df["date_ordinal"] = df["earnings_date"].map(lambda d: d.toordinal())
    df["batch_id"]     = (df["date_ordinal"] // group_window_days)

    signals_records = []

    for batch_id, batch in df.groupby("batch_id"):
        if len(batch) < 2:
            # With only 1 stock in batch, still generate signal if it's extreme
            batch = batch.copy()
            batch["rank"] = 1
            score = batch.iloc[0]["composite_score"]
            if score > 0.2:
                signal = 1
            elif score < -0.2:
                signal = -1
            else:
                signal = 0

            signals_records.append({
                "ticker":           batch.iloc[0]["ticker"],
                "earnings_date":    batch.iloc[0]["earnings_date"],
                "composite_score":  score,
                "signal":           signal,
                "rank":             1,
                "batch_size":       1,
                "position_size":    1.0 / max(n_long, 1),
                "batch_id":         batch_id,
            })
            continue

        batch = batch.copy()
        batch["rank"] = batch["composite_score"].rank(ascending=False, method="first").astype(int)
        batch_size    = len(batch)

        for _, row in batch.iterrows():
            rank  = int(row["rank"])
            score = float(row["composite_score"])

            if rank <= n_long:
                signal = 1
                pos_size = 1.0 / n_long
            elif rank > batch_size - n_short:
                signal = -1
                pos_size = 1.0 / n_short
            else:
                signal = 0
                pos_size = 0.0

            signals_records.append({
                "ticker":          row["ticker"],
                "earnings_date":   row["earnings_date"],
                "composite_score": score,
                "signal":          signal,
                "rank":            rank,
                "batch_size":      batch_size,
                "position_size":   pos_size,
                "batch_id":        batch_id,
            })

    signals_df = pd.DataFrame(signals_records)

    # Add quintile based on cross-sectional ranking across ALL events
    if len(signals_df) > 0:
        signals_df["quintile"] = pd.qcut(
            signals_df["composite_score"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

    return signals_df.sort_values("earnings_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge signals with event study results
# ---------------------------------------------------------------------------

def merge_signals_with_events(
    signals_df: pd.DataFrame,
    events_scored: pd.DataFrame,
    event_study_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join signals, scored events, and event study CARs into one master frame.

    Parameters
    ----------
    signals_df          : pd.DataFrame  Output of generate_signals().
    events_scored       : pd.DataFrame  Events with FinBERT scores.
    event_study_results : pd.DataFrame  Output of run_event_study().

    Returns
    -------
    pd.DataFrame  Master frame with all columns merged.
    """
    # Start from signals
    merged = signals_df.merge(
        events_scored[[
            "ticker", "earnings_date",
            "positive_prob", "negative_prob", "neutral_prob",
            "forward_5d_return", "exit_date",
        ]],
        on=["ticker", "earnings_date"],
        how="left",
    )

    # Merge event study results
    merged = merged.merge(
        event_study_results[[
            "ticker", "event_date",
            "alpha", "beta", "r_squared",
            "car_m1_p5", "car_0_p5", "car_m1_p1", "bhar_20", "valid",
        ]],
        left_on=["ticker", "earnings_date"],
        right_on=["ticker", "event_date"],
        how="left",
    )

    return merged.sort_values("earnings_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Information Coefficient (IC) computation
# ---------------------------------------------------------------------------

def compute_ic(
    signals_df: pd.DataFrame,
    score_col: str = "composite_score",
    return_col: str = "forward_5d_return",
) -> dict:
    """
    Compute the Information Coefficient: Spearman rank correlation between
    composite_score and forward_5d_return.

    IC > 0 implies the sentiment model has predictive value.
    IC > 0.05 is considered meaningful in practice; IC > 0.10 is strong.

    Parameters
    ----------
    signals_df  : pd.DataFrame  Master signals frame.
    score_col   : str           Column with composite sentiment score.
    return_col  : str           Column with realised forward return.

    Returns
    -------
    dict with keys: ic_spearman, ic_pearson, n_obs, p_value.
    """
    from scipy import stats as scipy_stats

    valid = signals_df[[score_col, return_col]].dropna()
    n     = len(valid)

    if n < 10:
        return {"ic_spearman": np.nan, "ic_pearson": np.nan, "n_obs": n, "p_value": np.nan}

    spearman_r, spearman_p = scipy_stats.spearmanr(valid[score_col], valid[return_col])
    pearson_r,  pearson_p  = scipy_stats.pearsonr(valid[score_col],  valid[return_col])

    return {
        "ic_spearman": float(spearman_r),
        "ic_pearson":  float(pearson_r),
        "n_obs":       n,
        "p_value":     float(spearman_p),
        "significant": spearman_p < 0.05,
    }


# ---------------------------------------------------------------------------
# Quintile CAR summary
# ---------------------------------------------------------------------------

def quintile_car_summary(master_df: pd.DataFrame, car_col: str = "car_m1_p5") -> pd.DataFrame:
    """
    Compute mean CAR by sentiment quintile to verify monotonic relationship.

    Parameters
    ----------
    master_df : pd.DataFrame  Output of merge_signals_with_events().
    car_col   : str           CAR column to summarise.

    Returns
    -------
    pd.DataFrame  Mean CAR, std, count, t-stat per quintile.
    """
    from scipy import stats as scipy_stats

    records = []
    for q in range(1, 6):
        subset = master_df.loc[master_df["quintile"] == q, car_col].dropna()
        if len(subset) < 2:
            continue
        t_stat, p_val = scipy_stats.ttest_1samp(subset, popmean=0)
        records.append({
            "quintile":  q,
            "n":         len(subset),
            "mean_car":  float(subset.mean()),
            "std_car":   float(subset.std()),
            "t_stat":    float(t_stat),
            "p_value":   float(p_val),
        })

    return pd.DataFrame(records).set_index("quintile")
