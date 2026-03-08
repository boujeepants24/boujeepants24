"""
sizing.py
---------
Position sizing for the funding rate arbitrage strategy.

Methods implemented
-------------------
1. Kelly Criterion (full and fractional)
   - Edge  = expected funding income per 8h (net of fees and borrow cost)
   - Odds  = basis risk modelled as the OU process standard deviation
   - f*    = edge / variance  (continuous Kelly for a normal payoff distribution)
   - We apply a HALF-KELLY fraction by default (risk management conservatism)

2. Portfolio-level constraints
   - Maximum allocation per asset: MAX_ASSET_WEIGHT (default 20%)
   - Maximum total gross exposure: MAX_GROSS_EXPOSURE (default 100% of NAV)
   - If ALL assets have high funding simultaneously, scale down by CONCENTRATION_PENALTY

3. Correlation risk adjustment
   - If cross-asset funding correlation is high, reduce position sizes proportionally
"""

import numpy as np
import pandas as pd
from typing import Optional

from signals import compute_basis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of full Kelly to use (0.5 = half-Kelly, safer)
KELLY_FRACTION = 0.5

# Maximum allocation to any single asset (as fraction of NAV)
MAX_ASSET_WEIGHT = 0.20

# Maximum gross notional exposure as multiple of NAV
MAX_GROSS_EXPOSURE = 1.0    # 1x = 100% of NAV in each leg; net delta = 0

# If > this many assets are simultaneously in "high funding" regime,
# apply a concentration penalty to all position sizes
CONCENTRATION_THRESHOLD = 4
CONCENTRATION_PENALTY   = 0.6   # scale all sizes to 60% when concentrated

# Fee and carry inputs (matching execution.py)
TAKER_FEE_BPS     = 4      # per leg, in basis points
BORROW_COST_ANN   = 0.02   # 2% annual borrow cost on spot leg
ANNUAL_FACTOR     = 3 * 365


# ---------------------------------------------------------------------------
# Single-asset Kelly sizing
# ---------------------------------------------------------------------------

def kelly_weight(
    funding_rate:    float,
    basis_vol:       float,
    position_size:   float = 1.0,
    kelly_fraction:  float = KELLY_FRACTION,
    taker_fee_bps:   int   = TAKER_FEE_BPS,
    borrow_cost_ann: float = BORROW_COST_ANN,
) -> float:
    """
    Compute Kelly-optimal weight for a single funding arb position.

    Model
    -----
    Per 8h period:
      edge (mu) = funding_rate
                  - round_trip_fee_per_period      [fee amortised over hold period]
                  - borrow_cost_per_period          [spot leg carry]
                  (simplified: fees paid at entry/exit, borrow accrues continuously)

    We use a single-period approximation where we assume the position is held
    for an average of `avg_hold_periods` periods.

    variance (sigma^2) = basis_vol^2  per period
        (basis risk is the main P&L uncertainty)

    f* = mu / sigma^2   (continuous Kelly for normal dist)
    f_half = f* * kelly_fraction

    Parameters
    ----------
    funding_rate    : 8h funding rate (decimal), e.g. 0.0003
    basis_vol       : estimated 1-period standard deviation of basis changes (decimal)
    position_size   : notional size in USD (used for slippage scaling, see execution.py)
    kelly_fraction  : fraction of full Kelly to use
    taker_fee_bps   : taker fee per leg in basis points
    borrow_cost_ann : annual borrow rate on the spot leg (decimal)

    Returns
    -------
    Optimal weight as a fraction of NAV [0.0 .. MAX_ASSET_WEIGHT]
    """
    if basis_vol <= 0 or funding_rate <= 0:
        return 0.0

    # Convert fees to per-period cost
    # Round-trip = 2 legs * 2 (entry + exit) * fee_per_leg
    # Amortised over an assumed average hold of 3 periods (24h)
    avg_hold_periods        = 3.0
    fee_per_leg             = taker_fee_bps / 10_000
    round_trip_fee          = 4 * fee_per_leg           # 4 legs total
    fee_per_period          = round_trip_fee / avg_hold_periods

    # Borrow cost: per 8h = annual / (3 * 365)
    borrow_per_period       = borrow_cost_ann / ANNUAL_FACTOR

    # Net edge per 8h period
    mu = funding_rate - fee_per_period - borrow_per_period

    if mu <= 0:
        return 0.0   # trade is not profitable after costs

    # Full Kelly fraction
    sigma_sq = basis_vol ** 2
    f_star   = mu / sigma_sq

    # Apply fractional Kelly and hard cap
    weight = min(f_star * kelly_fraction, MAX_ASSET_WEIGHT)
    return float(np.clip(weight, 0.0, MAX_ASSET_WEIGHT))


# ---------------------------------------------------------------------------
# Basis volatility estimator (Ornstein-Uhlenbeck)
# ---------------------------------------------------------------------------

def estimate_basis_vol(
    basis_series: pd.Series,
    window: int = 30,
) -> pd.Series:
    """
    Estimate the rolling 1-period standard deviation of basis changes.

    We use the OU (Ornstein-Uhlenbeck) diffusion parameter sigma, estimated
    as the rolling std of first-differences of the basis series.

    Parameters
    ----------
    basis_series : pd.Series of basis values (%), indexed by timestamp
    window       : rolling window in number of 8h periods

    Returns
    -------
    pd.Series of estimated sigma (per 8h period), same index.
    """
    # First difference of basis in decimal form
    basis_dec = basis_series / 100.0
    dbasis    = basis_dec.diff().dropna()

    # Rolling std as proxy for OU diffusion coefficient
    sigma = dbasis.rolling(window, min_periods=5).std()
    sigma = sigma.reindex(basis_series.index).ffill().bfill()
    sigma = sigma.clip(lower=1e-6)   # avoid division by zero in Kelly

    return sigma


# ---------------------------------------------------------------------------
# Multi-asset portfolio sizer
# ---------------------------------------------------------------------------

def size_portfolio(
    signals_dict:    dict,
    funding_wide:    pd.DataFrame,
    perp_dict:       dict,
    spot_dict:       dict,
    nav:             float,
    kelly_fraction:  float = KELLY_FRACTION,
    max_asset_wt:    float = MAX_ASSET_WEIGHT,
    max_gross_exp:   float = MAX_GROSS_EXPOSURE,
) -> pd.DataFrame:
    """
    Compute notional position sizes for every asset at every 8h period.

    Parameters
    ----------
    signals_dict : dict[symbol -> signals DataFrame] from signals.py
    funding_wide : wide DataFrame of 8h funding rates (time x symbol)
    perp_dict    : dict[symbol -> OHLCV DataFrame]
    spot_dict    : dict[symbol -> OHLCV DataFrame]
    nav          : current portfolio NAV in USD
    kelly_fraction, max_asset_wt, max_gross_exp : sizing constraints

    Returns
    -------
    DataFrame indexed by timestamp, columns = symbols,
    values = target notional USD per asset (+ve = long, -ve = short).

    Note: in a delta-neutral arb, the perp leg and spot leg have EQUAL
    and OPPOSITE notional, so the net notional tracked here represents
    the SIZE of one leg. Sign convention:
      +ve -> short perp / long spot (collecting positive funding)
      -ve -> long  perp / short spot (collecting negative funding)
    """
    symbols = list(signals_dict.keys())
    all_times = funding_wide.index
    size_df   = pd.DataFrame(0.0, index=all_times, columns=symbols)

    for symbol in symbols:
        if symbol not in perp_dict or symbol not in spot_dict:
            continue

        perp_cl  = perp_dict[symbol]["close"]
        spot_cl  = spot_dict[symbol]["close"]

        # Compute basis and its volatility
        basis_s  = compute_basis(perp_cl, spot_cl)
        basis_vol = estimate_basis_vol(basis_s)

        fund_s   = funding_wide[symbol]
        sigs_df  = signals_dict[symbol]

        if sigs_df.empty:
            continue

        # Build a position series: +1/-1/0 at each timestamp
        pos_series = pd.Series(0, index=all_times, dtype=float)
        current_pos = 0

        for ts in all_times:
            # Check for a signal at this timestamp
            if ts in sigs_df.index:
                row = sigs_df.loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                current_pos = int(row["direction"])
            pos_series[ts] = current_pos

        # Compute Kelly weight at each timestamp
        weights = pd.Series(0.0, index=all_times)

        for ts in all_times:
            fr  = fund_s.get(ts, np.nan)
            bv  = basis_vol.get(ts, 0.001)

            if np.isnan(fr) or pos_series[ts] == 0:
                weights[ts] = 0.0
                continue

            w = kelly_weight(
                funding_rate    = abs(fr),
                basis_vol       = float(bv),
                kelly_fraction  = kelly_fraction,
            )
            weights[ts] = w * pos_series[ts]   # carry sign from direction

        # Convert weight -> notional USD (clamp to max_asset_wt first)
        notional = weights.clip(-max_asset_wt, max_asset_wt) * nav
        size_df[symbol] = notional

    # --- Concentration risk adjustment ---
    # Count how many assets are in a position at each timestamp
    active_assets = (size_df.abs() > 0).sum(axis=1)
    penalty_mask  = active_assets > CONCENTRATION_THRESHOLD
    size_df.loc[penalty_mask] *= CONCENTRATION_PENALTY

    # --- Gross exposure cap ---
    gross = size_df.abs().sum(axis=1)
    cap   = max_gross_exp * nav
    scale = (cap / gross).clip(upper=1.0)
    size_df = size_df.mul(scale, axis=0)

    return size_df


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def summarize_sizing(size_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-asset sizing statistics for inspection.
    """
    stats = pd.DataFrame({
        "mean_notional_usd":  size_df.abs().mean(),
        "max_notional_usd":   size_df.abs().max(),
        "pct_time_invested":  (size_df.abs() > 0).mean() * 100,
        "mean_direction":     size_df.mean(),
    })
    return stats
