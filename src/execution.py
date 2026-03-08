"""
execution.py
------------
Execution model: realistically accounts for all transaction costs.

Components
----------
1. Taker fees     : 4 bps per leg (Binance USDT-margined perps + spot)
2. Slippage       : 2 bps per $1M notional (market impact, scales linearly)
3. Funding income : credited every 8h based on mark price and position direction
4. Borrow cost    : 2% annualized on spot leg notional (if borrowing to hedge)

All returns are in USD.

Fee summary for a FULL round-trip (open + close, both perp and spot legs):
  - 4 legs * 4 bps = 16 bps total fees
  - 4 legs * slippage = 4 * (notional / 1e6) * 2 bps
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAKER_FEE_BPS    = 4            # per leg, in basis points
SLIPPAGE_BPS_PER_1M = 2         # bps of slippage per $1M notional per leg
BORROW_COST_ANN  = 0.02         # annual borrow rate on spot notional
ANNUAL_FACTOR    = 3 * 365      # 8h periods per year


# ---------------------------------------------------------------------------
# Data class for a single trade event
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """
    Immutable record of a single execution event (open or close of one leg).

    Attributes
    ----------
    timestamp   : UTC datetime of the trade
    symbol      : asset, e.g. "BTCUSDT"
    leg         : "perp" or "spot"
    side        : "buy" or "sell"
    price       : execution price (after slippage)
    notional    : USD notional traded (positive)
    fee_usd     : fee paid in USD
    slippage_usd: slippage cost in USD
    """
    timestamp:     pd.Timestamp
    symbol:        str
    leg:           str
    side:          str
    price:         float
    notional:      float
    fee_usd:       float
    slippage_usd:  float


@dataclass
class FundingPayment:
    """
    Record of one funding payment received or paid.
    """
    timestamp:       pd.Timestamp
    symbol:          str
    funding_rate:    float      # 8h rate (decimal)
    perp_notional:   float      # notional of perp position (signed: +ve = long perp)
    payment_usd:     float      # received if positive, paid if negative


# ---------------------------------------------------------------------------
# Fee and slippage computations
# ---------------------------------------------------------------------------

def compute_fee(notional_usd: float, taker_fee_bps: int = TAKER_FEE_BPS) -> float:
    """
    Taker fee for one leg.
    fee = notional * fee_rate
    """
    return notional_usd * (taker_fee_bps / 10_000)


def compute_slippage(
    notional_usd: float,
    slippage_bps_per_1m: float = SLIPPAGE_BPS_PER_1M,
) -> float:
    """
    Market impact / slippage cost for one leg.
    slippage = notional * (notional / 1_000_000) * (bps / 10_000)

    This models linear price impact: larger trades move the market more.
    """
    impact_rate = (notional_usd / 1_000_000) * (slippage_bps_per_1m / 10_000)
    return notional_usd * impact_rate


def total_open_cost(
    notional_usd: float,
    taker_fee_bps: int   = TAKER_FEE_BPS,
    slippage_bps_per_1m: float = SLIPPAGE_BPS_PER_1M,
) -> float:
    """
    Total friction cost to OPEN one leg (fee + slippage) in USD.
    Called for both the perp leg and the spot leg.
    """
    return compute_fee(notional_usd, taker_fee_bps) + compute_slippage(notional_usd, slippage_bps_per_1m)


def round_trip_cost(
    notional_usd: float,
    taker_fee_bps: int   = TAKER_FEE_BPS,
    slippage_bps_per_1m: float = SLIPPAGE_BPS_PER_1M,
    num_legs: int = 2,
) -> float:
    """
    Full round-trip cost (open + close, num_legs legs each way).
    Default: 2 legs (perp + spot) opened and closed = 4 executions total.
    """
    one_way = total_open_cost(notional_usd, taker_fee_bps, slippage_bps_per_1m)
    return one_way * num_legs * 2   # open and close, both legs


# ---------------------------------------------------------------------------
# Execution price with slippage
# ---------------------------------------------------------------------------

def fill_price(
    mid_price: float,
    notional_usd: float,
    side: str,
    slippage_bps_per_1m: float = SLIPPAGE_BPS_PER_1M,
) -> float:
    """
    Return the fill price after slippage.

    For a buy  order, price moves UP   by the slippage rate.
    For a sell order, price moves DOWN by the slippage rate.

    slippage_rate = (notional / 1e6) * slippage_bps_per_1m / 10_000
    """
    slippage_rate = (notional_usd / 1_000_000) * (slippage_bps_per_1m / 10_000)
    if side == "buy":
        return mid_price * (1 + slippage_rate)
    else:
        return mid_price * (1 - slippage_rate)


# ---------------------------------------------------------------------------
# Trade execution helpers
# ---------------------------------------------------------------------------

def execute_open(
    timestamp:   pd.Timestamp,
    symbol:      str,
    direction:   int,        # +1 = short perp / long spot; -1 = long perp / short spot
    notional:    float,      # USD notional for EACH leg
    perp_price:  float,
    spot_price:  float,
    taker_fee_bps: int   = TAKER_FEE_BPS,
    slippage_bps_per_1m: float = SLIPPAGE_BPS_PER_1M,
) -> tuple[list[TradeRecord], float]:
    """
    Simulate opening a delta-neutral position.

    direction = +1  -> SHORT perp at perp_price, LONG spot at spot_price
    direction = -1  -> LONG  perp at perp_price, SHORT spot at spot_price

    Returns
    -------
    (list of 2 TradeRecord objects,  total_cost_usd)
    """
    records = []
    total_cost = 0.0

    # --- Perp leg ---
    perp_side = "sell" if direction == +1 else "buy"
    perp_fill = fill_price(perp_price, notional, perp_side, slippage_bps_per_1m)
    perp_fee  = compute_fee(notional, taker_fee_bps)
    perp_slip = compute_slippage(notional, slippage_bps_per_1m)

    records.append(TradeRecord(
        timestamp    = timestamp,
        symbol       = symbol,
        leg          = "perp",
        side         = perp_side,
        price        = perp_fill,
        notional     = notional,
        fee_usd      = perp_fee,
        slippage_usd = perp_slip,
    ))
    total_cost += perp_fee + perp_slip

    # --- Spot leg ---
    spot_side = "buy" if direction == +1 else "sell"
    spot_fill = fill_price(spot_price, notional, spot_side, slippage_bps_per_1m)
    spot_fee  = compute_fee(notional, taker_fee_bps)
    spot_slip = compute_slippage(notional, slippage_bps_per_1m)

    records.append(TradeRecord(
        timestamp    = timestamp,
        symbol       = symbol,
        leg          = "spot",
        side         = spot_side,
        price        = spot_fill,
        notional     = notional,
        fee_usd      = spot_fee,
        slippage_usd = spot_slip,
    ))
    total_cost += spot_fee + spot_slip

    return records, total_cost


def execute_close(
    timestamp:   pd.Timestamp,
    symbol:      str,
    direction:   int,        # direction of the EXISTING position being closed
    notional:    float,
    perp_price:  float,
    spot_price:  float,
    taker_fee_bps: int   = TAKER_FEE_BPS,
    slippage_bps_per_1m: float = SLIPPAGE_BPS_PER_1M,
) -> tuple[list[TradeRecord], float]:
    """
    Simulate closing a delta-neutral position (reverse of open).
    """
    # Closing = open in opposite direction
    return execute_open(
        timestamp, symbol, -direction, notional,
        perp_price, spot_price,
        taker_fee_bps, slippage_bps_per_1m,
    )


# ---------------------------------------------------------------------------
# Funding income computation
# ---------------------------------------------------------------------------

def compute_funding_payment(
    timestamp:      pd.Timestamp,
    symbol:         str,
    funding_rate:   float,    # 8h rate (decimal); positive -> longs pay shorts
    perp_notional:  float,    # signed: +ve = long perp, -ve = short perp
    perp_mark_price: float,
) -> FundingPayment:
    """
    Compute the funding payment for one 8h settlement.

    Mechanics (Binance):
      payment = funding_rate * contract_value
      If funding_rate > 0: longs pay shorts
        -> short perp RECEIVES: abs(notional) * funding_rate
        -> long  perp PAYS:    abs(notional) * funding_rate
      If funding_rate < 0: shorts pay longs
        -> logic inverts

    We compute from the perspective of the POSITION HOLDER:
      payment_usd = -sign(perp_notional) * funding_rate * abs(perp_notional)

    So:
      Short perp (perp_notional < 0), positive funding -> payment_usd > 0 (receive)
      Long  perp (perp_notional > 0), positive funding -> payment_usd < 0 (pay)

    Parameters
    ----------
    perp_notional : signed USD notional of the perp position
    """
    position_sign = np.sign(perp_notional)
    payment_usd   = -position_sign * funding_rate * abs(perp_notional)

    return FundingPayment(
        timestamp      = timestamp,
        symbol         = symbol,
        funding_rate   = funding_rate,
        perp_notional  = perp_notional,
        payment_usd    = float(payment_usd),
    )


# ---------------------------------------------------------------------------
# Borrow cost computation
# ---------------------------------------------------------------------------

def compute_borrow_cost(
    spot_notional: float,
    borrow_cost_ann: float = BORROW_COST_ANN,
) -> float:
    """
    8h borrow cost on the spot leg notional.

    borrow_per_period = spot_notional * (annual_rate / annual_factor)
    """
    return spot_notional * (borrow_cost_ann / ANNUAL_FACTOR)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def trade_records_to_df(records: list[TradeRecord]) -> pd.DataFrame:
    """Convert a list of TradeRecord objects to a tidy DataFrame."""
    if not records:
        return pd.DataFrame()
    rows = [
        {
            "timestamp":    r.timestamp,
            "symbol":       r.symbol,
            "leg":          r.leg,
            "side":         r.side,
            "price":        r.price,
            "notional":     r.notional,
            "fee_usd":      r.fee_usd,
            "slippage_usd": r.slippage_usd,
            "total_cost":   r.fee_usd + r.slippage_usd,
        }
        for r in records
    ]
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def funding_payments_to_df(payments: list[FundingPayment]) -> pd.DataFrame:
    """Convert a list of FundingPayment objects to a tidy DataFrame."""
    if not payments:
        return pd.DataFrame()
    rows = [
        {
            "timestamp":     p.timestamp,
            "symbol":        p.symbol,
            "funding_rate":  p.funding_rate,
            "perp_notional": p.perp_notional,
            "payment_usd":   p.payment_usd,
        }
        for p in payments
    ]
    return pd.DataFrame(rows).set_index("timestamp").sort_index()
