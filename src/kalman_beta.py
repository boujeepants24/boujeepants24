"""
kalman_beta.py
--------------
Online Kalman filter for estimating time-varying market beta (and alpha) for
each stock relative to SPY.

State vector  x = [beta, alpha]   (2 x 1)
Observation   y = R_stock         (scalar)
Design vector H = [R_spy, 1]      (1 x 2)

Dynamics (random walk on beta and alpha):
  x_{t} = x_{t-1} + w_t,    w_t ~ N(0, Q)

Measurement:
  y_t = H_t @ x_t + v_t,    v_t ~ N(0, R)

This is a standard Kalman filter implemented from scratch with numpy.
No scipy, no filterpy — just numpy.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------ #
#  Kalman filter state for one stock                                  #
# ------------------------------------------------------------------ #

@dataclass
class KalmanState:
    """Holds the current mean and covariance of the state estimate."""
    x: np.ndarray          # (2,) state vector [beta, alpha]
    P: np.ndarray          # (2, 2) state covariance


def make_initial_state(
    init_beta: float = 1.0,
    init_alpha: float = 0.0,
    init_var: float = 1.0,
) -> KalmanState:
    """Diffuse prior: start with beta=1, alpha=0, large variance."""
    x = np.array([init_beta, init_alpha], dtype=float)
    P = np.eye(2) * init_var
    return KalmanState(x=x, P=P)


# ------------------------------------------------------------------ #
#  Single-step update                                                 #
# ------------------------------------------------------------------ #

def kalman_update(
    state: KalmanState,
    r_stock: float,
    r_spy: float,
    Q: np.ndarray,
    R: float,
) -> KalmanState:
    """
    One Kalman predict-update cycle.

    Parameters
    ----------
    state   : current KalmanState
    r_stock : observed stock log return at time t
    r_spy   : observed SPY   log return at time t
    Q       : (2, 2) process noise covariance
    R       : measurement noise variance (scalar)

    Returns
    -------
    Updated KalmanState with new x and P.
    """
    # Observation design vector H = [r_spy, 1]
    H = np.array([r_spy, 1.0])

    # --- Predict ---
    x_pred = state.x                     # random walk: E[x_t] = x_{t-1}
    P_pred = state.P + Q                 # P_t|t-1 = P_{t-1} + Q

    # --- Innovation ---
    y_hat = H @ x_pred                   # predicted return
    innov = r_stock - y_hat              # innovation (scalar)
    S = H @ P_pred @ H + R              # innovation variance (scalar)

    # --- Kalman gain ---
    K = (P_pred @ H) / S                 # (2,) vector

    # --- Update ---
    x_new = x_pred + K * innov
    P_new = (np.eye(2) - np.outer(K, H)) @ P_pred

    return KalmanState(x=x_new, P=P_new)


# ------------------------------------------------------------------ #
#  Full online estimation for a single stock                          #
# ------------------------------------------------------------------ #

def estimate_betas_single(
    r_stock: np.ndarray,
    r_spy: np.ndarray,
    process_noise_beta: float = 1e-4,
    process_noise_alpha: float = 1e-6,
    measurement_noise: float = 1e-3,
    init_var: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Kalman filter on one stock's full return series.

    Parameters
    ----------
    r_stock, r_spy : 1D arrays of aligned daily log returns (same length)
    process_noise_beta  : variance of beta random walk per day
    process_noise_alpha : variance of alpha random walk per day
    measurement_noise   : variance of observation noise

    Returns
    -------
    betas  : (T,) array of filtered beta estimates
    alphas : (T,) array of filtered alpha estimates
    """
    T = len(r_stock)
    assert len(r_spy) == T, "r_stock and r_spy must have the same length"

    Q = np.diag([process_noise_beta, process_noise_alpha])
    R = measurement_noise

    state = make_initial_state(init_var=init_var)
    betas  = np.empty(T)
    alphas = np.empty(T)

    for t in range(T):
        state = kalman_update(state, r_stock[t], r_spy[t], Q, R)
        betas[t]  = state.x[0]
        alphas[t] = state.x[1]

    return betas, alphas


# ------------------------------------------------------------------ #
#  Batch estimation for the full universe                             #
# ------------------------------------------------------------------ #

def estimate_all_betas(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    process_noise_beta: float = 1e-4,
    process_noise_alpha: float = 1e-6,
    measurement_noise: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate time-varying Kalman beta and alpha for every stock in the universe.

    Parameters
    ----------
    returns     : (T, N) DataFrame of daily log returns for universe stocks
    spy_returns : (T,) Series of SPY daily log returns

    Returns
    -------
    beta_df  : (T, N) DataFrame of Kalman beta estimates
    alpha_df : (T, N) DataFrame of Kalman alpha estimates
    """
    print("Running Kalman filter for all tickers ...")

    # Align
    common_idx = returns.index.intersection(spy_returns.index)
    ret = returns.loc[common_idx].copy()
    spy = spy_returns.loc[common_idx].values

    beta_dict  = {}
    alpha_dict = {}

    for ticker in ret.columns:
        r_stock = ret[ticker].values
        # Handle any remaining NaNs by substituting 0 (treated as no info)
        r_stock_clean = np.where(np.isnan(r_stock), 0.0, r_stock)

        b, a = estimate_betas_single(
            r_stock_clean,
            spy,
            process_noise_beta=process_noise_beta,
            process_noise_alpha=process_noise_alpha,
            measurement_noise=measurement_noise,
        )
        beta_dict[ticker]  = b
        alpha_dict[ticker] = a

    beta_df  = pd.DataFrame(beta_dict, index=ret.index)
    alpha_df = pd.DataFrame(alpha_dict, index=ret.index)

    print(f"  Kalman betas estimated for {len(ret.columns)} tickers over {len(ret)} days")
    return beta_df, alpha_df


# ------------------------------------------------------------------ #
#  Convenience: get latest beta at a given date                       #
# ------------------------------------------------------------------ #

def get_betas_at(beta_df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Return the most recent Kalman beta estimate on or before `date`."""
    idx = beta_df.index.get_indexer([date], method="pad")[0]
    if idx < 0:
        return pd.Series(np.nan, index=beta_df.columns)
    return beta_df.iloc[idx]
