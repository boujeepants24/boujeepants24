"""
backtest.py
-----------
Event-driven backtesting engine for the funding rate arbitrage strategy.

Architecture
------------
- Processes events in chronological order at 8-hour intervals (aligned to funding).
- At each timestep:
    1. Receive funding payment on all open positions
    2. Check exit signals; close positions if triggered
    3. Check entry signals; open new positions if risk gate approves
    4. Record account state

Account state tracked
---------------------
  cash            : USD cash balance (starts at initial_nav)
  perp_positions  : dict[symbol -> signed USD notional]
  spot_positions  : dict[symbol -> signed USD notional]
  unrealized_pnl  : mark-to-market on open positions
  realized_pnl    : cumulative closed-trade P&L
  funding_income  : cumulative funding received
  fee_cost        : cumulative fees paid
  borrow_cost     : cumulative borrow cost on spot legs

NAV = cash + unrealized_pnl  (realized already reflected in cash)

P&L attribution (per period):
  funding_income_period  += funding payments received
  basis_pnl_period       += change in unrealized basis P&L
  fee_cost_period        -= fees and slippage on executions
  borrow_cost_period     -= carry on spot positions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from execution import (
    execute_open, execute_close,
    compute_funding_payment,
    compute_borrow_cost,
    TradeRecord, FundingPayment,
    trade_records_to_df, funding_payments_to_df,
)
from signals import ENTRY_THRESHOLD, EXIT_THRESHOLD, BASIS_STOP_PCT
from risk import (
    pre_trade_check, fit_ou_process, compute_var_cvar,
    check_liquidation_risk, MAX_LEVERAGE,
)


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    """Mutable account state, updated at each 8h step."""
    cash:             float
    nav:              float = 0.0
    perp_positions:   dict  = field(default_factory=dict)   # symbol -> signed notional
    spot_positions:   dict  = field(default_factory=dict)   # symbol -> signed notional
    unrealized_pnl:   float = 0.0
    realized_pnl:     float = 0.0
    funding_income:   float = 0.0
    fee_cost:         float = 0.0
    borrow_cost_acc:  float = 0.0

    # Entry prices for open positions (to compute unrealized P&L)
    entry_perp_price: dict  = field(default_factory=dict)   # symbol -> entry price
    entry_spot_price: dict  = field(default_factory=dict)
    entry_direction:  dict  = field(default_factory=dict)   # symbol -> +1/-1


@dataclass
class PeriodRecord:
    """Snapshot of one 8h period for the ledger."""
    timestamp:           pd.Timestamp
    nav:                 float
    cash:                float
    realized_pnl:        float
    unrealized_pnl:      float
    funding_income_cum:  float
    fee_cost_cum:        float
    borrow_cost_cum:     float
    period_funding:      float
    period_basis_pnl:    float
    period_fees:         float
    period_borrow:       float
    gross_notional:      float
    leverage:            float
    n_positions:         int


# ---------------------------------------------------------------------------
# Main backtester
# ---------------------------------------------------------------------------

class FundingArbBacktester:
    """
    Event-driven backtester for the funding rate arbitrage strategy.

    Usage
    -----
    bt = FundingArbBacktester(
        funding_wide=..., perp_dict=..., spot_dict=...,
        initial_nav=1_000_000,
    )
    results = bt.run()
    """

    def __init__(
        self,
        funding_wide:    pd.DataFrame,
        perp_dict:       dict,
        spot_dict:       dict,
        initial_nav:     float  = 1_000_000.0,
        entry_threshold: float  = ENTRY_THRESHOLD,
        exit_threshold:  float  = EXIT_THRESHOLD,
        basis_stop_pct:  float  = BASIS_STOP_PCT,
        ou_lookback:     int    = 90,    # periods for OU estimation
        verbose:         bool   = False,
    ):
        self.funding_wide     = funding_wide
        self.perp_dict        = perp_dict
        self.spot_dict        = spot_dict
        self.initial_nav      = initial_nav
        self.entry_threshold  = entry_threshold
        self.exit_threshold   = exit_threshold
        self.basis_stop_pct   = basis_stop_pct
        self.ou_lookback      = ou_lookback
        self.verbose          = verbose

        # Will be populated by run()
        self.ledger:      list[PeriodRecord] = []
        self.trades:      list[TradeRecord]  = []
        self.payments:    list[FundingPayment] = []
        self.account:     Optional[AccountState] = None

    # ------------------------------------------------------------------
    # Helper: current prices at timestamp
    # ------------------------------------------------------------------

    def _price(self, symbol: str, ts: pd.Timestamp, leg: str) -> Optional[float]:
        """Return close price for given symbol/leg at timestamp, or None."""
        try:
            src = self.perp_dict[symbol] if leg == "perp" else self.spot_dict[symbol]
            if ts in src.index:
                return float(src.loc[ts, "close"])
            # Fall back to nearest prior price
            prior = src.index[src.index <= ts]
            if len(prior) > 0:
                return float(src.loc[prior[-1], "close"])
        except (KeyError, IndexError):
            pass
        return None

    # ------------------------------------------------------------------
    # Helper: OU params for a symbol
    # ------------------------------------------------------------------

    def _get_ou_params(self, symbol: str, ts: pd.Timestamp) -> dict:
        """Fit OU process to recent basis series for the given symbol."""
        try:
            perp_cl = self.perp_dict[symbol]["close"]
            spot_cl = self.spot_dict[symbol]["close"]
            # Use data up to (not including) current timestamp
            idx     = perp_cl.index[perp_cl.index < ts][-self.ou_lookback:]
            if len(idx) < 10:
                return {"kappa": 0.01, "theta": 0.0, "sigma": 0.005,
                        "half_life_periods": 69, "half_life_days": 23}
            basis_s = (perp_cl.loc[idx] - spot_cl.loc[idx]) / spot_cl.loc[idx] * 100
            return fit_ou_process(basis_s)
        except Exception:
            return {"kappa": 0.01, "theta": 0.0, "sigma": 0.005,
                    "half_life_periods": 69, "half_life_days": 23}

    # ------------------------------------------------------------------
    # Unrealized P&L
    # ------------------------------------------------------------------

    def _compute_unrealized(self, ts: pd.Timestamp) -> float:
        """
        Mark-to-market unrealized P&L across all open positions.

        For a delta-neutral position (short perp / long spot):
          unrealized_pnl = (entry_perp - current_perp) * qty_perp
                         + (current_spot - entry_spot) * qty_spot

        We work in notional space so:
          perp pnl = direction * notional * (entry_perp - current_perp) / entry_perp
          spot pnl = direction * notional * (current_spot - entry_spot) / entry_spot
        """
        total = 0.0
        acct  = self.account
        for sym in list(acct.perp_positions.keys()):
            ep   = acct.entry_perp_price.get(sym)
            es   = acct.entry_spot_price.get(sym)
            cp   = self._price(sym, ts, "perp")
            cs   = self._price(sym, ts, "spot")
            dirn = acct.entry_direction.get(sym, 0)
            notl = abs(acct.perp_positions.get(sym, 0))

            if None in (ep, es, cp, cs) or notl == 0:
                continue

            # direction=+1 means short perp / long spot
            perp_pnl = -dirn * notl * (cp - ep) / ep
            spot_pnl =  dirn * notl * (cs - es) / es
            total   += perp_pnl + spot_pnl

        return total

    # ------------------------------------------------------------------
    # Process one 8h period
    # ------------------------------------------------------------------

    def _process_period(
        self,
        ts:         pd.Timestamp,
        prev_unreal: float,
    ) -> PeriodRecord:
        acct         = self.account
        period_fund  = 0.0
        period_fees  = 0.0
        period_borrow= 0.0

        # ---------------------------------------------------------------
        # 1. Funding payments on all open perp positions
        # ---------------------------------------------------------------
        for sym in list(acct.perp_positions.keys()):
            perp_notl = acct.perp_positions[sym]
            if perp_notl == 0:
                continue
            fr  = self.funding_wide.get(sym, pd.Series(dtype=float)).get(ts, np.nan)
            mkt = self._price(sym, ts, "perp")
            if np.isnan(fr) or mkt is None:
                continue

            fp = compute_funding_payment(ts, sym, fr, perp_notl, mkt)
            acct.cash           += fp.payment_usd
            acct.funding_income += fp.payment_usd
            period_fund         += fp.payment_usd
            self.payments.append(fp)

            # Borrow cost on spot leg (accrues continuously, debited every period)
            spot_notl = abs(acct.spot_positions.get(sym, 0))
            bc        = compute_borrow_cost(spot_notl)
            acct.cash           -= bc
            acct.borrow_cost_acc += bc
            period_borrow        += bc

        # ---------------------------------------------------------------
        # 2. Check exits on open positions
        # ---------------------------------------------------------------
        for sym in list(acct.perp_positions.keys()):
            perp_notl = acct.perp_positions[sym]
            if perp_notl == 0:
                continue

            dirn   = acct.entry_direction.get(sym, 0)
            fr     = self.funding_wide.get(sym, pd.Series(dtype=float)).get(ts, np.nan)
            cp     = self._price(sym, ts, "perp")
            cs     = self._price(sym, ts, "spot")
            eb     = acct.entry_perp_price.get(sym, cp)
            es_pr  = acct.entry_spot_price.get(sym, cs)

            if None in (cp, cs) or np.isnan(fr):
                continue

            current_basis = (cp - cs) / cs * 100 if cs else 0
            entry_basis   = (eb - es_pr) / es_pr * 100 if es_pr else 0
            basis_move    = (current_basis - entry_basis) * dirn

            should_exit   = (
                abs(fr) < self.exit_threshold            # funding decayed
                or basis_move > self.basis_stop_pct      # basis stop
            )

            if should_exit:
                notl   = abs(perp_notl)
                trades, cost = execute_close(ts, sym, dirn, notl, cp, cs)
                self.trades.extend(trades)
                acct.cash    -= cost
                acct.fee_cost += cost
                period_fees  += cost

                # Compute realized P&L
                perp_pnl = -dirn * notl * (cp - eb) / eb
                spot_pnl =  dirn * notl * (cs - es_pr) / es_pr
                realized  = perp_pnl + spot_pnl - cost
                acct.cash         += realized
                acct.realized_pnl += realized

                # Clear position
                del acct.perp_positions[sym]
                del acct.spot_positions[sym]
                acct.entry_perp_price.pop(sym, None)
                acct.entry_spot_price.pop(sym, None)
                acct.entry_direction.pop(sym, None)

                if self.verbose:
                    print(f"  EXIT  {sym} at {ts} | realized={realized:.2f}")

        # ---------------------------------------------------------------
        # 3. Check entries (only if we have capacity)
        # ---------------------------------------------------------------
        for sym in self.funding_wide.columns:
            if sym in acct.perp_positions:
                continue   # already in a position

            fr = self.funding_wide.get(sym, pd.Series(dtype=float)).get(ts, np.nan)
            if np.isnan(fr) or abs(fr) < self.entry_threshold:
                continue

            cp = self._price(sym, ts, "perp")
            cs = self._price(sym, ts, "spot")
            if cp is None or cs is None:
                continue

            dirn     = 1 if fr > 0 else -1
            nav_now  = acct.cash + self._compute_unrealized(ts)
            ou_prms  = self._get_ou_params(sym, ts)
            basis_p  = (cp - cs) / cs * 100 if cs else 0

            # Propose notional: flat 15% of NAV per asset (simple base sizing)
            proposed = nav_now * 0.15

            gate = pre_trade_check(
                symbol            = sym,
                proposed_notional = proposed,
                current_positions = {k: abs(v) for k, v in acct.perp_positions.items()},
                nav               = nav_now,
                funding_rate      = fr,
                basis_pct         = basis_p,
                ou_params         = ou_prms,
            )

            if not gate["approved"] or gate["adj_notional"] < 1_000:
                continue

            notl = gate["adj_notional"]
            trades, cost = execute_open(ts, sym, dirn, notl, cp, cs)
            self.trades.extend(trades)
            acct.cash     -= cost
            acct.fee_cost += cost
            period_fees   += cost

            # Record open position
            acct.perp_positions[sym]  = -dirn * notl  # short perp = -notional
            acct.spot_positions[sym]  =  dirn * notl  # long  spot = +notional
            acct.entry_perp_price[sym] = cp
            acct.entry_spot_price[sym] = cs
            acct.entry_direction[sym]  = dirn

            if self.verbose:
                print(f"  ENTER {sym} at {ts} | dir={dirn} notl={notl:.0f} fr={fr*100:.4f}%")

        # ---------------------------------------------------------------
        # 4. Mark-to-market and record state
        # ---------------------------------------------------------------
        unreal       = self._compute_unrealized(ts)
        basis_pnl_p  = unreal - prev_unreal
        nav_now      = acct.cash + unreal

        acct.unrealized_pnl = unreal
        acct.nav            = nav_now

        gross_notl = sum(abs(v) for v in acct.perp_positions.values()) * 2
        leverage   = gross_notl / max(nav_now, 1.0)
        n_pos      = len(acct.perp_positions)

        rec = PeriodRecord(
            timestamp          = ts,
            nav                = nav_now,
            cash               = acct.cash,
            realized_pnl       = acct.realized_pnl,
            unrealized_pnl     = unreal,
            funding_income_cum = acct.funding_income,
            fee_cost_cum       = acct.fee_cost,
            borrow_cost_cum    = acct.borrow_cost_acc,
            period_funding     = period_fund,
            period_basis_pnl   = basis_pnl_p,
            period_fees        = period_fees,
            period_borrow      = period_borrow,
            gross_notional     = gross_notl,
            leverage           = leverage,
            n_positions        = n_pos,
        )
        self.ledger.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Run the backtest
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full backtest over the available data range.

        Returns
        -------
        dict with keys:
          "ledger"      : DataFrame of per-period account state
          "trades"      : DataFrame of all trade records
          "payments"    : DataFrame of all funding payments
          "final_nav"   : final portfolio value
          "total_return": (final_nav - initial_nav) / initial_nav
        """
        print("Starting backtest...")
        self.ledger   = []
        self.trades   = []
        self.payments = []

        self.account = AccountState(cash=self.initial_nav)

        # Common time index: use funding_wide index as the event clock
        all_times = self.funding_wide.index.sort_values()
        n_periods = len(all_times)

        prev_unreal = 0.0

        for i, ts in enumerate(all_times):
            if i % 500 == 0:
                nav_now = self.account.cash + prev_unreal
                n_pos   = len(self.account.perp_positions)
                print(f"  [{i}/{n_periods}] {ts.date()} | NAV={nav_now:,.0f} | Positions={n_pos}")

            rec          = self._process_period(ts, prev_unreal)
            prev_unreal  = rec.unrealized_pnl

        print("Backtest complete.")

        ledger_df  = pd.DataFrame([r.__dict__ for r in self.ledger]).set_index("timestamp")
        trades_df  = trade_records_to_df(self.trades)
        payments_df= funding_payments_to_df(self.payments)

        final_nav    = self.account.nav
        total_return = (final_nav - self.initial_nav) / self.initial_nav

        print(f"  Initial NAV : {self.initial_nav:>15,.2f}")
        print(f"  Final NAV   : {final_nav:>15,.2f}")
        print(f"  Total Return: {total_return*100:>+.2f}%")

        return {
            "ledger":       ledger_df,
            "trades":       trades_df,
            "payments":     payments_df,
            "final_nav":    final_nav,
            "total_return": total_return,
            "account":      self.account,
        }
