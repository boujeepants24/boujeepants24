"""
event_study.py
--------------
Event study methodology: Market Model estimation, Abnormal Returns,
Cumulative Abnormal Returns (CAR), and Buy-and-Hold Abnormal Returns (BHAR).

Methodology reference:
  MacKinlay, A.C. (1997). Event studies in economics and finance.
  Journal of Economic Literature, 35(1), 13-39.

Market Model:
  R_i,t = alpha_i + beta_i * R_m,t + epsilon_i,t

  Estimation window: [-130, -11] trading days relative to event (120 days),
  ending 10 days before the event to avoid contamination.

  R_i = log return of stock i
  R_m = log return of market proxy (SPY)

Abnormal Return:
  AR_i,t = R_i,t - (alpha_i + beta_i * R_m,t)

Cumulative Abnormal Return over window [tau1, tau2]:
  CAR_i(tau1, tau2) = sum_{t=tau1}^{tau2} AR_i,t

Statistical test:
  t-stat = mean(CAR) / (std(CAR) / sqrt(N))
  Under H0: mean(CAR) = 0

BHAR (Buy-and-Hold Abnormal Return) over T days:
  BHAR_i = prod(1 + R_i,t) - prod(1 + E[R_i,t])
  where expected return uses the market model.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Estimation window: use 120 trading days, ending ESTIMATION_GAP before event
ESTIMATION_WINDOW  = 120
ESTIMATION_GAP     = 10   # days before event to end estimation (avoid contamination)

# Event window for CAR: look at -5 to +20 days around event
EVENT_WINDOW_PRE   = 5    # days before event
EVENT_WINDOW_POST  = 20   # days after event

# Primary CAR measure for trading signal
CAR_WINDOW         = (-1, 5)   # CAR(-1, +5) for 5-day hold
BHAR_WINDOW        = 20        # BHAR over 20 days post-event


# ---------------------------------------------------------------------------
# Market Model Estimation
# ---------------------------------------------------------------------------

def estimate_market_model(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    event_date: pd.Timestamp,
    estimation_window: int = ESTIMATION_WINDOW,
    estimation_gap: int = ESTIMATION_GAP,
) -> dict:
    """
    Estimate alpha and beta using OLS on the pre-event estimation window.

    Parameters
    ----------
    stock_returns   : pd.Series  Daily log returns of the stock.
    market_returns  : pd.Series  Daily log returns of SPY.
    event_date      : Timestamp  The earnings announcement date.
    estimation_window: int       Number of trading days to use for OLS.
    estimation_gap  : int        Days before event to end the estimation window.

    Returns
    -------
    dict with keys:
        alpha      : float  Intercept from OLS.
        beta       : float  Market beta from OLS.
        r_squared  : float  R^2 of the OLS fit.
        n_obs      : int    Actual number of observations used.
        valid      : bool   Whether estimation succeeded with enough data.
    """
    from sklearn.linear_model import LinearRegression

    # Align both series on common index
    common_idx    = stock_returns.index.intersection(market_returns.index)
    stock_aligned  = stock_returns.loc[common_idx]
    market_aligned = market_returns.loc[common_idx]

    # Find position of event date in the aligned index
    all_dates = common_idx.sort_values()
    event_pos = all_dates.searchsorted(event_date)

    # End of estimation window: event_pos - estimation_gap
    est_end_pos   = event_pos - estimation_gap
    est_start_pos = est_end_pos - estimation_window

    if est_start_pos < 0 or est_end_pos <= 0:
        return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0, "n_obs": 0, "valid": False}

    est_dates  = all_dates[est_start_pos:est_end_pos]
    n_obs      = len(est_dates)

    if n_obs < 30:  # Need minimum data for reliable OLS
        return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0, "n_obs": n_obs, "valid": False}

    R_i = stock_aligned.loc[est_dates].values
    R_m = market_aligned.loc[est_dates].values

    # Remove any NaN pairs
    mask = ~(np.isnan(R_i) | np.isnan(R_m))
    R_i  = R_i[mask]
    R_m  = R_m[mask]

    if len(R_i) < 30:
        return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0, "n_obs": len(R_i), "valid": False}

    # OLS via sklearn for robustness
    reg = LinearRegression(fit_intercept=True)
    reg.fit(R_m.reshape(-1, 1), R_i)

    alpha     = float(reg.intercept_)
    beta      = float(reg.coef_[0])
    r_squared = float(reg.score(R_m.reshape(-1, 1), R_i))

    return {
        "alpha":     alpha,
        "beta":      beta,
        "r_squared": r_squared,
        "n_obs":     len(R_i),
        "valid":     True,
    }


# ---------------------------------------------------------------------------
# Abnormal Return computation
# ---------------------------------------------------------------------------

def compute_abnormal_returns(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    event_date: pd.Timestamp,
    alpha: float,
    beta: float,
    pre_days: int = EVENT_WINDOW_PRE,
    post_days: int = EVENT_WINDOW_POST,
) -> pd.DataFrame:
    """
    Compute actual returns, expected returns, and abnormal returns
    over the event window.

    Parameters
    ----------
    stock_returns  : pd.Series  Daily log returns of stock.
    market_returns : pd.Series  Daily log returns of SPY.
    event_date     : Timestamp  Earnings announcement date.
    alpha, beta    : float      Market model parameters.
    pre_days       : int        Days before event to include.
    post_days      : int        Days after event to include.

    Returns
    -------
    pd.DataFrame with columns:
        date, event_day, actual_return, expected_return, abnormal_return.
    """
    common_idx  = stock_returns.index.intersection(market_returns.index)
    all_dates   = common_idx.sort_values()
    event_pos   = all_dates.searchsorted(event_date)

    start_pos   = max(0, event_pos - pre_days)
    end_pos     = min(len(all_dates), event_pos + post_days + 1)

    window_dates = all_dates[start_pos:end_pos]

    records = []
    for date in window_dates:
        event_day = int(all_dates.searchsorted(date) - event_pos)

        if date not in stock_returns.index or date not in market_returns.index:
            continue

        R_i = float(stock_returns.loc[date])
        R_m = float(market_returns.loc[date])

        expected = alpha + beta * R_m
        abnormal = R_i - expected

        records.append({
            "date":            date,
            "event_day":       event_day,
            "actual_return":   R_i,
            "expected_return": expected,
            "abnormal_return": abnormal,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CAR and BHAR
# ---------------------------------------------------------------------------

def compute_car(
    abnormal_returns_df: pd.DataFrame,
    tau1: int,
    tau2: int,
) -> float:
    """
    Compute Cumulative Abnormal Return over window [tau1, tau2].

    CAR(tau1, tau2) = sum of AR_t for t in [tau1, tau2].

    Parameters
    ----------
    abnormal_returns_df : pd.DataFrame  Output of compute_abnormal_returns().
    tau1 : int  Start of window (relative to event day, can be negative).
    tau2 : int  End of window (inclusive).

    Returns
    -------
    float  CAR value. NaN if insufficient data.
    """
    mask = (
        (abnormal_returns_df["event_day"] >= tau1) &
        (abnormal_returns_df["event_day"] <= tau2)
    )
    subset = abnormal_returns_df.loc[mask, "abnormal_return"]

    if len(subset) == 0:
        return np.nan

    return float(subset.sum())


def compute_bhar(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    event_date: pd.Timestamp,
    alpha: float,
    beta: float,
    horizon: int = BHAR_WINDOW,
) -> float:
    """
    Compute Buy-and-Hold Abnormal Return over `horizon` days post-event.

    BHAR = product(1 + R_i,t) - product(1 + E[R_i,t])  for t in [0, horizon-1]

    This is the geometric (compound) version of abnormal return, more
    appropriate for longer holding periods.

    Parameters
    ----------
    horizon : int  Number of trading days after the event.

    Returns
    -------
    float  BHAR value. NaN if insufficient data.
    """
    common_idx  = stock_returns.index.intersection(market_returns.index)
    all_dates   = common_idx.sort_values()
    event_pos   = all_dates.searchsorted(event_date)

    end_pos     = min(len(all_dates), event_pos + horizon + 1)
    window_dates = all_dates[event_pos:end_pos]

    if len(window_dates) < 2:
        return np.nan

    actual_rets   = []
    expected_rets = []

    for date in window_dates:
        if date not in stock_returns.index or date not in market_returns.index:
            continue
        R_i  = float(stock_returns.loc[date])
        R_m  = float(market_returns.loc[date])
        R_e  = alpha + beta * R_m
        actual_rets.append(R_i)
        expected_rets.append(R_e)

    if len(actual_rets) < 2:
        return np.nan

    # Convert log returns to simple returns before compounding
    bh_actual   = np.prod([np.exp(r) for r in actual_rets]) - 1.0
    bh_expected = np.prod([np.exp(r) for r in expected_rets]) - 1.0

    return float(bh_actual - bh_expected)


# ---------------------------------------------------------------------------
# Full event study for a single event
# ---------------------------------------------------------------------------

def run_single_event_study(
    ticker: str,
    event_date: pd.Timestamp,
    stock_returns: pd.Series,
    market_returns: pd.Series,
    pre_days: int = EVENT_WINDOW_PRE,
    post_days: int = EVENT_WINDOW_POST,
) -> dict:
    """
    Full event study pipeline for one earnings event.

    Returns
    -------
    dict with keys:
        ticker, event_date, alpha, beta, r_squared, n_obs, valid,
        car_m1_p5   : CAR(-1, +5)
        car_0_p5    : CAR(0, +5)
        car_m1_p1   : CAR(-1, +1) -- announcement effect
        bhar_20     : BHAR over 20 days
        ar_series   : pd.DataFrame  (full abnormal return series)
    """
    model_params = estimate_market_model(
        stock_returns=stock_returns,
        market_returns=market_returns,
        event_date=event_date,
    )

    alpha = model_params["alpha"]
    beta  = model_params["beta"]

    ar_df = compute_abnormal_returns(
        stock_returns=stock_returns,
        market_returns=market_returns,
        event_date=event_date,
        alpha=alpha,
        beta=beta,
        pre_days=pre_days,
        post_days=post_days,
    )

    car_m1_p5  = compute_car(ar_df, tau1=-1, tau2=5)
    car_0_p5   = compute_car(ar_df, tau1=0,  tau2=5)
    car_m1_p1  = compute_car(ar_df, tau1=-1, tau2=1)
    bhar_20    = compute_bhar(
        stock_returns=stock_returns,
        market_returns=market_returns,
        event_date=event_date,
        alpha=alpha,
        beta=beta,
        horizon=BHAR_WINDOW,
    )

    return {
        "ticker":     ticker,
        "event_date": event_date,
        **model_params,
        "car_m1_p5":  car_m1_p5,
        "car_0_p5":   car_0_p5,
        "car_m1_p1":  car_m1_p1,
        "bhar_20":    bhar_20,
        "ar_series":  ar_df,
    }


# ---------------------------------------------------------------------------
# Batch event study across all events
# ---------------------------------------------------------------------------

def run_event_study(
    events: pd.DataFrame,
    returns: pd.DataFrame,
    market_col: str = "SPY",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full event study for all earnings events.

    Parameters
    ----------
    events     : pd.DataFrame  Must have columns: ticker, earnings_date.
    returns    : pd.DataFrame  Daily log returns; columns include all tickers + market_col.
    market_col : str           Column name for the market proxy return.
    verbose    : bool          Print progress.

    Returns
    -------
    results_df : pd.DataFrame
        One row per event with CAR, BHAR, market model params.
    ar_panel   : dict
        ticker -> event_date -> abnormal return series (for plotting).
    """
    if market_col not in returns.columns:
        raise ValueError(f"Market column '{market_col}' not found in returns DataFrame.")

    market_returns = returns[market_col]
    ar_panel       = {}
    records        = []

    total = len(events)
    for i, (_, row) in enumerate(events.iterrows()):
        ticker      = row["ticker"]
        event_date  = row["earnings_date"]

        if verbose and i % 100 == 0:
            print(f"[event_study] Processing event {i+1}/{total}: {ticker} @ {event_date.date()}")

        if ticker not in returns.columns:
            continue

        stock_returns = returns[ticker]

        result = run_single_event_study(
            ticker=ticker,
            event_date=event_date,
            stock_returns=stock_returns,
            market_returns=market_returns,
        )

        ar_df = result.pop("ar_series")

        # Store AR series keyed by (ticker, event_date)
        ar_panel[(ticker, event_date)] = ar_df

        records.append(result)

    results_df = pd.DataFrame(records)
    print(f"[event_study] Completed. Valid events: {results_df['valid'].sum()} / {len(results_df)}")
    return results_df, ar_panel


# ---------------------------------------------------------------------------
# CAR significance testing
# ---------------------------------------------------------------------------

def test_car_significance(
    cars: pd.Series,
    label: str = "All events",
) -> dict:
    """
    Test if mean CAR is significantly different from zero (two-sided t-test).

    Parameters
    ----------
    cars  : pd.Series  CAR values (one per event).
    label : str        Label for logging.

    Returns
    -------
    dict with keys: n, mean_car, std_car, t_stat, p_value, significant_5pct.
    """
    cars_clean = cars.dropna()
    n          = len(cars_clean)

    if n < 2:
        return {"n": n, "mean_car": np.nan, "std_car": np.nan,
                "t_stat": np.nan, "p_value": np.nan, "significant_5pct": False}

    mean_car = float(cars_clean.mean())
    std_car  = float(cars_clean.std(ddof=1))
    t_stat   = float(stats.ttest_1samp(cars_clean, popmean=0).statistic)
    p_value  = float(stats.ttest_1samp(cars_clean, popmean=0).pvalue)

    result = {
        "label":           label,
        "n":               n,
        "mean_car":        mean_car,
        "std_car":         std_car,
        "t_stat":          t_stat,
        "p_value":         p_value,
        "significant_5pct": p_value < 0.05,
    }
    return result


def summarize_by_sentiment(
    events_with_scores: pd.DataFrame,
    results_df: pd.DataFrame,
    sentiment_col: str = "sentiment_label",
    car_col: str = "car_m1_p5",
) -> pd.DataFrame:
    """
    Summarize CAR statistics broken down by sentiment label.

    Parameters
    ----------
    events_with_scores : pd.DataFrame  Events with composite_score and sentiment_label.
    results_df         : pd.DataFrame  CAR results from run_event_study().
    sentiment_col      : str           Column with categorical sentiment label.
    car_col            : str           Which CAR metric to analyse.

    Returns
    -------
    pd.DataFrame  Summary table with t-stats and p-values per sentiment bucket.
    """
    # If car_col already present (pre-merged), use directly; else merge
    if car_col in events_with_scores.columns:
        merged = events_with_scores.dropna(subset=[car_col]).copy()
    else:
        merged = events_with_scores.merge(
            results_df[["ticker", "event_date", car_col]],
            left_on=["ticker", "earnings_date"],
            right_on=["ticker", "event_date"],
            how="inner",
        )

    summaries = []
    for label in ["positive", "neutral", "negative"]:
        subset = merged.loc[merged[sentiment_col] == label, car_col]
        stats_dict = test_car_significance(subset, label=label)
        summaries.append(stats_dict)

    # Also do overall
    stats_dict = test_car_significance(merged[car_col], label="all")
    summaries.append(stats_dict)

    return pd.DataFrame(summaries).set_index("label")
