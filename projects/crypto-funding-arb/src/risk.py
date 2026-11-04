"""
risk.py
-------
Risk management module for the funding rate arbitrage engine.

Covers:
  1. Delta neutrality tracking  -- net delta should stay near 0
  2. Basis risk (OU model)      -- model mean-reversion of basis spread
  3. Liquidation risk guard     -- enforce margin buffer, max leverage = 3x
  4. VaR / CVaR estimation      -- parametric and historical simulation
  5. Correlation risk           -- penalise concentrated cross-asset exposure
  6. Real-time risk checks      -- go/no-go gate before entering a trade
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LEVERAGE          = 3.0      # never exceed 3x gross notional / NAV
MARGIN_BUFFER_RATIO   = 0.20     # maintain at least 20% of margin as buffer
MAX_DELTA_EXPOSURE    = 0.05     # acceptable net delta as fraction of NAV (5%)
CORR_HIGH_THRESHOLD   = 0.70     # cross-asset funding correlation above this = correlated regime
CORR_REDUCTION_FACTOR = 0.50     # reduce sizes by 50% in correlated regime

VAR_CONFIDENCE        = 0.95
CVAR_CONFIDENCE       = 0.95


# ---------------------------------------------------------------------------
# Data class for risk snapshot
# ---------------------------------------------------------------------------

@dataclass
class RiskSnapshot:
    """
    Point-in-time risk metrics for the portfolio.

    delta_usd     : net USD delta (should be ~0 for delta-neutral)
    delta_pct_nav : net delta as % of NAV
    gross_notional: sum of |notional| across all positions
    leverage      : gross_notional / NAV
    var_95        : 1-day VaR at 95% confidence (USD, positive = loss)
    cvar_95       : 1-day CVaR at 95% (USD, positive = loss)
    max_basis_move: worst basis move in current lookback window
    is_safe       : True if all risk checks pass
    warnings      : list of warning strings
    """
    timestamp:       pd.Timestamp
    delta_usd:       float
    delta_pct_nav:   float
    gross_notional:  float
    leverage:        float
    var_95:          float
    cvar_95:         float
    max_basis_move:  float
    is_safe:         bool
    warnings:        list


# ---------------------------------------------------------------------------
# Delta tracking
# ---------------------------------------------------------------------------

def compute_net_delta(
    perp_positions: dict,   # symbol -> signed USD notional (+ = long)
    spot_positions: dict,   # symbol -> signed USD notional (+ = long)
) -> dict:
    """
    Compute net USD delta per asset and total.

    Delta-neutral: long spot + short perp -> delta = spot + perp ~= 0
    Any deviation represents un-hedged directional exposure.

    Returns
    -------
    dict with:
      per_asset  : dict[symbol -> net_delta_usd]
      total      : float, sum of all deltas
    """
    symbols    = set(perp_positions.keys()) | set(spot_positions.keys())
    per_asset  = {}
    for sym in symbols:
        p = perp_positions.get(sym, 0.0)
        s = spot_positions.get(sym, 0.0)
        per_asset[sym] = p + s   # net directional exposure

    return {"per_asset": per_asset, "total": sum(per_asset.values())}


def delta_ok(delta_usd: float, nav: float) -> bool:
    """Return True if net delta is within acceptable bounds."""
    return abs(delta_usd) <= MAX_DELTA_EXPOSURE * nav


# ---------------------------------------------------------------------------
# Basis risk: Ornstein-Uhlenbeck model
# ---------------------------------------------------------------------------

def fit_ou_process(basis_series: pd.Series) -> dict:
    """
    Fit an Ornstein-Uhlenbeck (mean-reverting) process to the basis series.

    The OU SDE:  dX_t = kappa*(theta - X_t)*dt + sigma*dW_t

    Discretised (Euler-Maruyama, dt = 1 period):
      X_{t+1} - X_t = kappa*(theta - X_t) + sigma * eps_t

    Estimated via OLS regression of (X_{t+1} - X_t) on X_t.

    Returns
    -------
    dict with keys:
      kappa  : mean-reversion speed (per 8h period)
      theta  : long-run mean of basis
      sigma  : volatility of basis innovations
      half_life_periods : -ln(2) / ln(1 - kappa)  [number of 8h periods]
      half_life_days    : half_life_periods / 3
    """
    x   = basis_series.dropna().values
    if len(x) < 10:
        return {"kappa": 0, "theta": 0, "sigma": 0, "half_life_periods": np.inf, "half_life_days": np.inf}

    y   = np.diff(x)        # X_{t+1} - X_t
    x_t = x[:-1]            # X_t

    # OLS: y = alpha + beta * x_t
    slope, intercept, _, _, _ = stats.linregress(x_t, y)

    # kappa = -beta  (mean-reversion rate, positive)
    kappa = float(-slope)
    theta = float(intercept / kappa) if kappa != 0 else 0.0

    # Sigma: std of residuals
    residuals = y - (intercept + slope * x_t)
    sigma     = float(np.std(residuals))

    # Half-life: time for deviation to decay to half its initial value
    if kappa > 0:
        hl_periods = np.log(2) / kappa
    else:
        hl_periods = np.inf
    hl_days = hl_periods / 3.0

    return {
        "kappa":               kappa,
        "theta":               theta,
        "sigma":               sigma,
        "half_life_periods":   hl_periods,
        "half_life_days":      hl_days,
    }


def basis_stress_scenario(ou_params: dict, n_sigma: float = 3.0) -> float:
    """
    Estimate the worst-case basis move (in %) over a 24h horizon (3 periods).

    Under the OU model the variance of the process over n_periods is approximately:
      Var(X_T) = sigma^2 * (1 - exp(-2*kappa*n)) / (2*kappa)

    We return n_sigma standard deviations of this as the stress scenario.
    """
    kappa  = ou_params.get("kappa",  0.01)
    sigma  = ou_params.get("sigma",  0.005)
    n_periods = 3

    if kappa > 0:
        variance = sigma**2 * (1 - np.exp(-2 * kappa * n_periods)) / (2 * kappa)
    else:
        variance = sigma**2 * n_periods

    stress = n_sigma * np.sqrt(variance)
    return float(stress)


# ---------------------------------------------------------------------------
# Liquidation risk
# ---------------------------------------------------------------------------

def check_liquidation_risk(
    gross_notional: float,
    nav:            float,
    margin_buffer:  float = MARGIN_BUFFER_RATIO,
    max_leverage:   float = MAX_LEVERAGE,
) -> dict:
    """
    Check whether the current leverage is within safe limits.

    Returns
    -------
    dict:
      leverage    : current leverage (gross_notional / nav)
      is_safe     : True if leverage <= max_leverage * (1 - margin_buffer)
      safe_limit  : maximum gross notional before approaching liquidation risk
    """
    leverage   = gross_notional / max(nav, 1.0)
    safe_limit = nav * max_leverage * (1 - margin_buffer)
    is_safe    = leverage <= max_leverage * (1 - margin_buffer)

    return {
        "leverage":   leverage,
        "is_safe":    is_safe,
        "safe_limit": safe_limit,
    }


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------

def compute_var_cvar(
    returns_series: pd.Series,
    confidence:     float = VAR_CONFIDENCE,
    method:         str   = "historical",
) -> tuple[float, float]:
    """
    Compute Value-at-Risk and Conditional VaR (Expected Shortfall).

    Parameters
    ----------
    returns_series : pd.Series of 8h P&L in USD
    confidence     : confidence level, e.g. 0.95
    method         : "historical" (empirical quantile) or "parametric" (normal approx)

    Returns
    -------
    (var_usd, cvar_usd) both expressed as positive loss amounts
    """
    r = returns_series.dropna()
    if len(r) < 10:
        return (0.0, 0.0)

    if method == "historical":
        var  = float(-np.percentile(r, (1 - confidence) * 100))
        tail = r[r <= -var]
        cvar = float(-tail.mean()) if len(tail) > 0 else var

    else:   # parametric (normal)
        mu    = r.mean()
        sigma = r.std()
        z     = stats.norm.ppf(1 - confidence)
        var   = float(-(mu + z * sigma))
        cvar  = float(-(mu - sigma * stats.norm.pdf(z) / (1 - confidence)))

    return max(var, 0.0), max(cvar, 0.0)


def rolling_var_cvar(
    returns_series: pd.Series,
    window:         int   = 90,    # 90 * 3 = 270 trading days
    confidence:     float = VAR_CONFIDENCE,
) -> pd.DataFrame:
    """
    Compute rolling historical VaR and CVaR over the return series.

    Returns
    -------
    DataFrame with columns ["var_95", "cvar_95"], same index as returns_series.
    """
    var_vals  = []
    cvar_vals = []

    for i in range(len(returns_series)):
        window_r = returns_series.iloc[max(0, i - window): i + 1]
        v, c     = compute_var_cvar(window_r, confidence)
        var_vals.append(v)
        cvar_vals.append(c)

    df = pd.DataFrame(
        {"var_95": var_vals, "cvar_95": cvar_vals},
        index=returns_series.index,
    )
    return df


# ---------------------------------------------------------------------------
# Correlation risk
# ---------------------------------------------------------------------------

def funding_correlation_matrix(
    funding_wide: pd.DataFrame,
    window:       int = 90,
) -> pd.DataFrame:
    """
    Compute rolling pairwise Pearson correlation of funding rates.

    A high average correlation means all assets are in the same regime,
    which reduces diversification and concentrates risk.
    """
    return funding_wide.rolling(window, min_periods=30).corr()


def mean_pairwise_correlation(
    funding_wide: pd.DataFrame,
    window:       int = 90,
) -> pd.Series:
    """
    Return the rolling mean pairwise correlation (off-diagonal) as a Series.
    Scalar per timestamp, useful for regime detection and position scaling.
    """
    results = {}
    data    = funding_wide.dropna(how="all")

    for i in range(window, len(data)):
        window_df = data.iloc[i - window: i]
        corr_mat  = window_df.corr()
        # Extract off-diagonal values
        n         = len(corr_mat)
        if n < 2:
            results[data.index[i]] = 0.0
            continue
        upper     = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        mean_corr = float(upper.stack().mean())
        results[data.index[i]] = mean_corr

    return pd.Series(results, name="mean_pairwise_corr")


def corr_size_adjustment(mean_corr: float) -> float:
    """
    Return a size multiplier based on average pairwise correlation.

    low corr (< 0.4) : full size (1.0)
    medium   (0.4-0.7): partial reduction
    high     (> 0.7) : CORR_REDUCTION_FACTOR
    """
    if mean_corr < 0.4:
        return 1.0
    elif mean_corr > CORR_HIGH_THRESHOLD:
        return CORR_REDUCTION_FACTOR
    else:
        # Linear interpolation
        t = (mean_corr - 0.4) / (CORR_HIGH_THRESHOLD - 0.4)
        return 1.0 - t * (1 - CORR_REDUCTION_FACTOR)


# ---------------------------------------------------------------------------
# Full risk check (pre-trade gate)
# ---------------------------------------------------------------------------

def pre_trade_check(
    symbol:          str,
    proposed_notional: float,
    current_positions: dict,   # existing notionals by symbol
    nav:             float,
    funding_rate:    float,
    basis_pct:       float,
    ou_params:       dict,
    mean_corr:       float = 0.0,
) -> dict:
    """
    Go/no-go risk gate before entering a new position.

    Checks:
      1. Leverage limit: would this trade push leverage above MAX_LEVERAGE?
      2. Basis stress: is the OU-predicted worst-case basis move acceptable?
      3. Correlation: is overall cross-asset correlation too high?

    Returns
    -------
    dict:
      approved    : bool
      reasons     : list of strings explaining rejections or approvals
      adj_notional: adjusted (possibly reduced) notional that passes all checks
    """
    reasons      = []
    approved     = True
    adj_notional = proposed_notional

    # Check 1: leverage
    current_gross = sum(abs(v) for v in current_positions.values())
    new_gross     = current_gross + proposed_notional * 2  # both legs
    leverage      = new_gross / max(nav, 1.0)

    if leverage > MAX_LEVERAGE:
        # Scale back to fit within limit
        available = max(0, nav * MAX_LEVERAGE - current_gross)
        adj_notional = available / 2   # half for each leg
        reasons.append(
            f"Leverage limit: adjusted from {proposed_notional:.0f} to {adj_notional:.0f}"
        )
        if adj_notional < proposed_notional * 0.1:
            approved = False
            reasons.append("Trade too small after leverage adjustment; rejected")

    # Check 2: basis stress
    stress = basis_stress_scenario(ou_params, n_sigma=3.0)
    if stress > 2.0:   # 2% worst-case basis move is too high
        reasons.append(f"High basis risk: OU 3-sigma stress = {stress:.2f}%")
        adj_notional *= 0.5
        reasons.append(f"Halved notional to {adj_notional:.0f} due to basis risk")

    # Check 3: correlation
    corr_adj = corr_size_adjustment(mean_corr)
    if corr_adj < 1.0:
        adj_notional *= corr_adj
        reasons.append(
            f"Correlation penalty {corr_adj:.2f}x applied; "
            f"mean pair corr = {mean_corr:.2f}"
        )

    if not reasons:
        reasons.append("All risk checks passed")

    return {
        "approved":     approved,
        "reasons":      reasons,
        "adj_notional": max(0.0, adj_notional),
    }


# ---------------------------------------------------------------------------
# Portfolio risk summary (post-period)
# ---------------------------------------------------------------------------

def portfolio_risk_summary(
    returns_series:  pd.Series,
    positions_df:    pd.DataFrame,
    funding_wide:    pd.DataFrame,
    nav_series:      pd.Series,
) -> dict:
    """
    Compute end-of-backtest risk statistics.

    Returns
    -------
    dict with VaR, CVaR, max leverage, avg delta, worst drawdown, etc.
    """
    var_95, cvar_95 = compute_var_cvar(returns_series, confidence=0.95)

    leverage_series = (positions_df.abs().sum(axis=1) * 2) / nav_series.clip(lower=1)
    max_leverage    = float(leverage_series.max())
    avg_leverage    = float(leverage_series.mean())

    # Daily delta series: abs sum of position notionals (should be ~0)
    avg_delta_pct   = float(
        (positions_df.sum(axis=1).abs() / nav_series.clip(lower=1)).mean() * 100
    )

    return {
        "var_95_usd":         var_95,
        "cvar_95_usd":        cvar_95,
        "max_leverage":       max_leverage,
        "avg_leverage":       avg_leverage,
        "avg_net_delta_pct":  avg_delta_pct,
    }
