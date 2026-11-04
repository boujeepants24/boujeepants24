"""
signals.py
----------
Signal generation for the funding rate arbitrage strategy.

Core idea:
  - When funding_rate > entry_threshold  -> SHORT perp / LONG spot (collect funding)
  - When funding_rate < -entry_threshold -> LONG perp  / SHORT spot (collect negative funding)
  - Hold until: rate decays below exit_threshold, OR basis moves adversely, OR stop-loss hit

The module also provides:
  - Funding rate autocorrelation analysis (persistence of high-funding regimes)
  - Basis computation: (perp_close - spot_close) / spot_close * 100
  - Annualized funding rate conversion
  - Regime detection (high vs normal funding environment)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Default threshold constants (all rates are in decimal, NOT percent)
# ---------------------------------------------------------------------------

# Binance funding rate is expressed as a fraction per 8-hour period.
# e.g., 0.0001 = 0.01% per 8h
ENTRY_THRESHOLD  = 0.0003   # 0.03% per 8h ~= 33% annualized   (go short perp / long spot)
EXIT_THRESHOLD   = 0.0001   # 0.01% per 8h ~= 11% annualized   (close position)
BASIS_STOP_PCT   = 1.0      # close if basis moves against us by >1% from entry
DELTA_STOP_PCT   = 2.0      # close if net delta PnL > 2% loss

# Annualization: 3 funding periods per day * 365 days
ANNUAL_FACTOR = 3 * 365


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """
    A single entry or exit signal at a given timestamp.

    direction : +1 = short perp / long spot (positive funding)
                -1 = long  perp / short spot (negative funding)
                 0 = no position / flat
    symbol    : asset symbol, e.g. "BTCUSDT"
    funding_rate      : raw 8h funding rate at signal time
    annualized_rate   : funding_rate * ANNUAL_FACTOR
    basis_pct         : (perp - spot) / spot * 100 at signal time
    signal_type       : "entry" or "exit"
    exit_reason       : None for entries; "funding_decay" | "basis_stop" | "stop_loss" for exits
    """
    timestamp:        pd.Timestamp
    symbol:           str
    direction:        int
    funding_rate:     float
    annualized_rate:  float
    basis_pct:        float
    signal_type:      str
    exit_reason:      Optional[str] = None


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def annualize_funding(funding_rate: pd.Series) -> pd.Series:
    """
    Convert 8-hour funding rate to annualized rate.

    funding_rate * 3 periods/day * 365 days/year
    """
    return funding_rate * ANNUAL_FACTOR


def compute_basis(perp_close: pd.Series, spot_close: pd.Series) -> pd.Series:
    """
    Basis = (perp_price - spot_price) / spot_price * 100  (expressed as %)

    Positive basis -> perp trades at premium -> shorts collect funding.
    Negative basis -> perp trades at discount -> longs collect negative funding.
    """
    return (perp_close - spot_close) / spot_close * 100


def compute_autocorrelation(funding_rates: pd.Series, max_lags: int = 10) -> pd.Series:
    """
    Compute autocorrelation of funding rates at lags 1..max_lags.

    High autocorrelation at lag-1 implies funding rates are persistent,
    which is one of the key prerequisites for this strategy to work.

    Returns
    -------
    pd.Series indexed by lag (1..max_lags), values = autocorrelation coefficient.
    """
    acf_values = {}
    for lag in range(1, max_lags + 1):
        acf_values[lag] = funding_rates.autocorr(lag=lag)
    return pd.Series(acf_values, name=funding_rates.name or "acf")


def detect_regime(
    funding_rates: pd.Series,
    high_thresh: float = ENTRY_THRESHOLD,
    window: int = 3,
) -> pd.Series:
    """
    Label each 8h period as 'high' or 'normal' funding regime.

    A period is labeled 'high' if the rolling mean of the ABSOLUTE funding
    rate over the last `window` periods exceeds high_thresh.

    Returns
    -------
    pd.Series of str: "high" or "normal", same index as input.
    """
    rolling_mean = funding_rates.abs().rolling(window, min_periods=1).mean()
    regime = pd.Series(
        np.where(rolling_mean >= high_thresh, "high", "normal"),
        index=funding_rates.index,
        name="regime",
    )
    return regime


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    funding_series: pd.Series,
    perp_close: pd.Series,
    spot_close: pd.Series,
    symbol: str,
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold: float  = EXIT_THRESHOLD,
    basis_stop_pct: float  = BASIS_STOP_PCT,
) -> list[Signal]:
    """
    Produce a chronological list of entry and exit Signal objects for one asset.

    Strategy logic (per 8h bar):
    -------------------------------------------------------
    When FLAT:
      - If |funding_rate| >= entry_threshold  -> ENTRY
        direction = +1 if funding_rate > 0 (short perp, long spot)
        direction = -1 if funding_rate < 0 (long perp, short spot)

    When IN POSITION:
      - EXIT if |funding_rate| < exit_threshold      (funding decayed)
      - EXIT if basis moved adversely > basis_stop_pct% from entry basis
        * short-perp position: adverse = basis INCREASES (perp premium widened)
        * long-perp  position: adverse = basis DECREASES (perp discount widened)

    Parameters
    ----------
    funding_series   : 8h funding rates (decimal), indexed by datetime
    perp_close       : perp closing prices, same index
    spot_close       : spot closing prices, same index
    symbol           : asset name for Signal objects
    entry_threshold  : minimum |funding_rate| to enter
    exit_threshold   : |funding_rate| below which we exit
    basis_stop_pct   : basis adverse move (%) that triggers an exit

    Returns
    -------
    list of Signal objects in chronological order
    """
    # Align all series on a common index
    idx = funding_series.index.intersection(perp_close.index).intersection(spot_close.index)
    funding = funding_series.reindex(idx)
    perp    = perp_close.reindex(idx)
    spot    = spot_close.reindex(idx)

    basis        = compute_basis(perp, spot)
    annualized   = annualize_funding(funding)

    signals      = []
    position     = 0       # 0 = flat, +1 = short-perp/long-spot, -1 = long-perp/short-spot
    entry_basis  = None    # basis at entry, used for basis stop

    for ts in idx:
        fr    = funding[ts]
        ann   = annualized[ts]
        bas   = basis[ts]

        if np.isnan(fr) or np.isnan(bas):
            continue

        # --- In position: check exits ---
        if position != 0:
            exit_reason = None

            # Exit condition 1: funding decayed
            if abs(fr) < exit_threshold:
                exit_reason = "funding_decay"

            # Exit condition 2: basis stop
            elif entry_basis is not None:
                basis_move = bas - entry_basis
                # Short-perp (direction=+1): bad if basis widens (more positive)
                # Long-perp  (direction=-1): bad if basis narrows (more negative)
                adverse_move = basis_move * position   # positive = adverse
                if adverse_move > basis_stop_pct:
                    exit_reason = "basis_stop"

            if exit_reason:
                signals.append(Signal(
                    timestamp       = ts,
                    symbol          = symbol,
                    direction       = 0,
                    funding_rate    = fr,
                    annualized_rate = ann,
                    basis_pct       = bas,
                    signal_type     = "exit",
                    exit_reason     = exit_reason,
                ))
                position    = 0
                entry_basis = None
            # If no exit, we remain in position (no new signal emitted - position persists)

        # --- Flat: check entry ---
        elif abs(fr) >= entry_threshold:
            direction = 1 if fr > 0 else -1
            signals.append(Signal(
                timestamp       = ts,
                symbol          = symbol,
                direction       = direction,
                funding_rate    = fr,
                annualized_rate = ann,
                basis_pct       = bas,
                signal_type     = "entry",
            ))
            position    = direction
            entry_basis = bas

    return signals


def signals_to_dataframe(signals: list[Signal]) -> pd.DataFrame:
    """
    Convert a list of Signal objects into a tidy DataFrame, indexed by timestamp.
    """
    if not signals:
        return pd.DataFrame()
    records = [
        {
            "timestamp":       s.timestamp,
            "symbol":          s.symbol,
            "direction":       s.direction,
            "funding_rate":    s.funding_rate,
            "annualized_rate": s.annualized_rate,
            "basis_pct":       s.basis_pct,
            "signal_type":     s.signal_type,
            "exit_reason":     s.exit_reason,
        }
        for s in signals
    ]
    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Multi-asset signal generation
# ---------------------------------------------------------------------------

def generate_all_signals(
    funding_wide: pd.DataFrame,
    perp_dict:    dict,
    spot_dict:    dict,
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold:  float = EXIT_THRESHOLD,
    basis_stop_pct:  float = BASIS_STOP_PCT,
) -> dict:
    """
    Run signal generation for every asset in the universe.

    Parameters
    ----------
    funding_wide : wide DataFrame (time x symbol) of 8h funding rates
    perp_dict    : dict[symbol -> OHLCV DataFrame with 'close' column]
    spot_dict    : dict[symbol -> OHLCV DataFrame with 'close' column]

    Returns
    -------
    dict[symbol -> signals DataFrame]
    """
    all_signals = {}
    for symbol in funding_wide.columns:
        if symbol not in perp_dict or symbol not in spot_dict:
            continue
        fund_s  = funding_wide[symbol].dropna()
        perp_cl = perp_dict[symbol]["close"]
        spot_cl = spot_dict[symbol]["close"]

        sigs = generate_signals(
            fund_s, perp_cl, spot_cl, symbol,
            entry_threshold, exit_threshold, basis_stop_pct,
        )
        all_signals[symbol] = signals_to_dataframe(sigs)
        print(f"  {symbol}: {len(sigs)} signals generated")

    return all_signals
