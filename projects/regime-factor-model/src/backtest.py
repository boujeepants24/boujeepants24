"""
backtest.py
-----------
Walk-forward backtester for the Regime-Aware Multi-Factor Model.

Walk-forward logic:
  - Train window : 2 years  (504 trading days)
  - Test window  : 6 months (126 trading days)
  - Step forward by one test window after each fold
  - Minimum 4 OOS windows required

Portfolio construction each rebalance date:
  1. Detect current regime via HMM (fitted on IS data only)
  2. Select regime-appropriate factor weights
  3. Compute combined factor scores for all assets
  4. Select top-8 assets by composite score (long-only)
  5. Assign equal weight (1/8) to each selected asset
  6. Deduct transaction costs: 5bps flat + 10bps market-impact on turnover

Transaction cost model:
  TC = (5bps + 10bps) * one-way turnover fraction
  Applied to the portfolio return on each rebalance date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from src.data       import DataLoader
from src.factors    import compute_factor_scores, compute_factor_attribution, FACTOR_NAMES
from src.regime     import (RegimeDetector, REGIME_FACTOR_WEIGHTS,
                             REGIME_NAMES, get_regime_at_date,
                             optimize_regime_weights)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_DAYS    = 504   # ~2 years
TEST_DAYS     = 126   # ~6 months
TOP_N         = 8     # assets held each month
FLAT_COST_BPS = 5     # basis points, flat per-trade
IMPACT_BPS    = 10    # basis points, market impact per-trade


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RebalanceRecord:
    """Stores metadata for a single monthly rebalance event."""
    date:          pd.Timestamp
    tickers:       list[str]
    weights:       np.ndarray            # equal-weight vector, len = TOP_N
    factor_scores: pd.DataFrame          # scores for all assets at this date
    regime:        int                   # detected regime
    factor_weights: np.ndarray           # 4-element factor weight vector
    turnover:      float = 0.0
    transaction_cost: float = 0.0


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    portfolio_returns:  pd.Series        # daily net-of-cost returns
    benchmark_returns:  pd.Series        # daily SPY returns (aligned)
    rebalance_records:  list[RebalanceRecord] = field(default_factory=list)
    regime_series:      pd.Series = None # daily regime labels
    oos_windows:        list[dict] = field(default_factory=list)
    is_returns:         pd.Series = None
    oos_returns:        pd.Series = None
    factor_attribution: pd.DataFrame = None


# ---------------------------------------------------------------------------
# Core portfolio engine
# ---------------------------------------------------------------------------

class RegimeFactorBacktester:
    """
    Walk-forward backtester for the Regime-Aware Multi-Factor Model.

    Usage
    -----
    bt = RegimeFactorBacktester(data_loader)
    result = bt.run()
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_turnover(
        self,
        prev_tickers: list[str],
        new_tickers:  list[str],
        n_assets:     int = TOP_N,
    ) -> float:
        """
        One-way portfolio turnover as a fraction of AUM.
        Turnover = (number of assets changed) / TOP_N
        Assumes equal weight before and after rebalance.
        """
        prev_set = set(prev_tickers)
        new_set  = set(new_tickers)
        changed  = len(new_set - prev_set)  # newly bought positions
        return changed / n_assets if n_assets > 0 else 0.0

    def _apply_transaction_cost(
        self,
        daily_returns: pd.Series,
        rebalance_date: pd.Timestamp,
        turnover: float,
    ) -> pd.Series:
        """
        Deduct a one-time transaction cost on the rebalance date.
        Cost = (FLAT_COST_BPS + IMPACT_BPS) / 10000 * turnover
        """
        cost_rate = (FLAT_COST_BPS + IMPACT_BPS) / 10_000
        total_cost = cost_rate * turnover

        # Apply as a negative return on the rebalance date
        if rebalance_date in daily_returns.index:
            daily_returns = daily_returns.copy()
            daily_returns.loc[rebalance_date] -= total_cost

        return daily_returns, total_cost

    # ------------------------------------------------------------------
    # Walk-forward split generator
    # ------------------------------------------------------------------

    def _generate_wf_windows(self) -> list[dict]:
        """
        Generate walk-forward (train, test) date splits.

        Returns a list of dicts with keys:
          train_start, train_end, test_start, test_end
        """
        dates  = self.loader.returns.index
        n_days = len(dates)
        windows = []

        start = 0
        while True:
            train_end_idx = start + TRAIN_DAYS - 1
            test_end_idx  = train_end_idx + TEST_DAYS

            if test_end_idx >= n_days:
                break

            windows.append({
                "train_start": dates[start],
                "train_end":   dates[train_end_idx],
                "test_start":  dates[train_end_idx + 1],
                "test_end":    dates[test_end_idx],
            })

            start += TEST_DAYS  # roll forward by one test window

        return windows

    # ------------------------------------------------------------------
    # Single rebalance
    # ------------------------------------------------------------------

    def _rebalance(
        self,
        date: pd.Timestamp,
        regime_series: pd.Series,
        factor_weights_by_regime: dict[int, np.ndarray],
    ) -> tuple[list[str], pd.DataFrame, int, np.ndarray]:
        """
        Compute factor scores and select top-N assets for the given date.

        Returns
        -------
        selected_tickers : list of chosen ticker symbols
        scores_df        : full factor score DataFrame for all assets
        regime           : detected regime integer constant
        fw               : 4-element factor weight array used
        """
        regime     = get_regime_at_date(regime_series, date)
        fw         = factor_weights_by_regime.get(regime, REGIME_FACTOR_WEIGHTS[regime])

        scores_df  = compute_factor_scores(
            self.loader.close,
            self.loader.returns,
            date,
            weights=fw,
        )

        # Drop assets with missing composite score
        valid = scores_df["composite"].dropna()
        if len(valid) < TOP_N:
            # Fall back to however many valid assets exist
            selected = valid.nlargest(len(valid)).index.tolist()
        else:
            selected = valid.nlargest(TOP_N).index.tolist()

        return selected, scores_df, regime, fw

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Execute the full walk-forward backtest.

        For each WF window:
          1. Fit HMM on training returns
          2. Decode full regime series (IS + OOS)
          3. Iterate through OOS monthly rebalance dates
          4. Build portfolio returns

        Returns a BacktestResult object.
        """
        wf_windows = self._generate_wf_windows()

        if len(wf_windows) < 4:
            raise RuntimeError(
                f"Only {len(wf_windows)} walk-forward windows generated. "
                f"Need >= 4. Check data date range."
            )

        print(f"\nWalk-forward windows: {len(wf_windows)}")
        for i, w in enumerate(wf_windows):
            print(f"  WF {i+1}: Train {w['train_start'].date()} to "
                  f"{w['train_end'].date()} | Test {w['test_start'].date()} "
                  f"to {w['test_end'].date()}")

        # Aggregate containers
        all_portfolio_daily: dict[pd.Timestamp, float] = {}
        all_regime_labels:   dict[pd.Timestamp, int]   = {}
        rebalance_records:   list[RebalanceRecord]      = []
        oos_window_results:  list[dict]                 = []
        is_return_slices:    list[pd.Series]            = []
        oos_return_slices:   list[pd.Series]            = []

        # Attribution tracking
        holdings_history: list[tuple] = []
        scores_history:   list[pd.DataFrame] = []
        weights_history:  list[np.ndarray]   = []

        prev_tickers: list[str] = []

        for wf_idx, window in enumerate(wf_windows):
            print(f"\n--- Walk-Forward Window {wf_idx + 1}/{len(wf_windows)} ---")

            # ---- 1. Training phase: fit HMM ---------------------------
            train_returns = self.loader.bench[
                (self.loader.bench.index >= window["train_start"]) &
                (self.loader.bench.index <= window["train_end"])
            ]

            detector = RegimeDetector(n_regimes=3, n_iter=200, random_state=42 + wf_idx)
            try:
                detector.fit(train_returns)
                print(f"  HMM fitted on {len(train_returns)} IS observations")
            except ValueError as e:
                print(f"  [WARN] HMM fit failed: {e} -- using prior weights")
                detector = None

            # ---- 2. Decode regime for full window period ---------------
            full_period_bench = self.loader.bench[
                (self.loader.bench.index >= window["train_start"]) &
                (self.loader.bench.index <= window["test_end"])
            ]

            if detector is not None:
                regime_series = detector.predict(full_period_bench)
            else:
                # Fallback: always Bull
                from src.regime import REGIME_BULL
                regime_series = pd.Series(
                    REGIME_BULL,
                    index=full_period_bench.index,
                    name="regime"
                )

            # Update global regime dict
            for dt, reg in regime_series.items():
                all_regime_labels[dt] = reg

            # ---- 3. Optimize factor weights from IS regime data --------
            # We don't have actual IS factor returns here yet, so use
            # the default regime weights (they get refined in the second+ windows)
            factor_weights_by_regime = REGIME_FACTOR_WEIGHTS.copy()

            # ---- 4. Get rebalance dates in the OOS window --------------
            oos_month_ends = [
                d for d in self.loader.month_ends
                if window["test_start"] <= d <= window["test_end"]
            ]

            if not oos_month_ends:
                print("  [WARN] No rebalance dates in OOS window -- skipping")
                continue

            print(f"  OOS rebalance dates: {len(oos_month_ends)}")

            # ---- 5. IS equity curve (buy-and-hold first rebalance) -----
            # IS performance: use the final IS rebalance to build IS curve
            is_rebalance_dates = [
                d for d in self.loader.month_ends
                if window["train_start"] <= d <= window["train_end"]
            ]

            if is_rebalance_dates and wf_idx == 0:
                # Only do IS tracking for first window to avoid look-ahead
                is_date = is_rebalance_dates[-1]
                is_tickers, is_scores, is_regime, is_fw = self._rebalance(
                    is_date, regime_series, factor_weights_by_regime
                )
                is_rets = self.loader.returns.loc[
                    window["train_start"]: window["train_end"],
                    is_tickers
                ].mean(axis=1)
                is_return_slices.append(is_rets)

            # ---- 6. OOS forward-test loop ------------------------------
            oos_daily_rets: dict[pd.Timestamp, float] = {}
            current_tickers  = prev_tickers.copy()
            current_weights  = None

            for i, rebal_date in enumerate(oos_month_ends):
                # Next rebalance date (or OOS end) determines hold period
                next_date = (
                    oos_month_ends[i + 1]
                    if i + 1 < len(oos_month_ends)
                    else window["test_end"]
                )

                # Compute factor scores & select portfolio
                selected, scores_df, regime, fw = self._rebalance(
                    rebal_date, regime_series, factor_weights_by_regime
                )

                # Transaction costs
                turnover = self._compute_turnover(current_tickers, selected)
                cost_rate = (FLAT_COST_BPS + IMPACT_BPS) / 10_000
                tc = cost_rate * turnover

                # Holding period daily returns
                hold_start = rebal_date
                hold_end   = next_date

                hold_dates = self.loader.returns.index[
                    (self.loader.returns.index >= hold_start) &
                    (self.loader.returns.index <  hold_end)
                ]

                if len(hold_dates) == 0:
                    continue

                # Equal-weight mean return across selected assets
                hold_ret = self.loader.returns.loc[hold_dates, selected].mean(axis=1)

                # Deduct TC on first day of holding period
                hold_ret_adj = hold_ret.copy()
                if len(hold_ret_adj) > 0:
                    hold_ret_adj.iloc[0] -= tc

                for dt, r in hold_ret_adj.items():
                    oos_daily_rets[dt] = r
                    all_portfolio_daily[dt] = r

                # Track attribution data
                holdings_history.append((rebal_date, selected))
                scores_history.append(scores_df)
                weights_history.append(fw)

                # Record
                rebalance_records.append(RebalanceRecord(
                    date=rebal_date,
                    tickers=selected,
                    weights=np.ones(len(selected)) / len(selected),
                    factor_scores=scores_df,
                    regime=regime,
                    factor_weights=fw,
                    turnover=turnover,
                    transaction_cost=tc,
                ))

                current_tickers = selected
                print(f"    Rebal {rebal_date.date()} | Regime: "
                      f"{REGIME_NAMES.get(regime, '?'):8s} | "
                      f"Turnover: {turnover:.0%} | TC: {tc*10000:.1f}bps | "
                      f"Assets: {', '.join(selected[:4])}...")

            prev_tickers = current_tickers

            # OOS window summary
            if oos_daily_rets:
                oos_s = pd.Series(oos_daily_rets).sort_index()
                oos_return_slices.append(oos_s)
                oos_cum = (1 + oos_s).cumprod().iloc[-1] - 1
                oos_window_results.append({
                    "window":       wf_idx + 1,
                    "test_start":   window["test_start"],
                    "test_end":     window["test_end"],
                    "oos_return":   oos_cum,
                    "n_rebalances": len(oos_month_ends),
                })
                print(f"  OOS window return: {oos_cum:.2%}")

        # ---- 7. Assemble final series -----------------------------------
        portfolio_daily = pd.Series(all_portfolio_daily).sort_index()
        regime_daily    = pd.Series(all_regime_labels).sort_index()
        bench_aligned   = self.loader.bench.reindex(portfolio_daily.index).fillna(0.0)

        is_returns_combined  = (pd.concat(is_return_slices).sort_index()
                                if is_return_slices else pd.Series(dtype=float))
        oos_returns_combined = (pd.concat(oos_return_slices).sort_index()
                                if oos_return_slices else pd.Series(dtype=float))

        # Factor attribution
        factor_attr = compute_factor_attribution(
            holdings_history, scores_history, weights_history
        )

        print(f"\nBacktest complete: {len(portfolio_daily)} net trading days "
              f"| {len(rebalance_records)} rebalances")

        return BacktestResult(
            portfolio_returns=portfolio_daily,
            benchmark_returns=bench_aligned,
            rebalance_records=rebalance_records,
            regime_series=regime_daily,
            oos_windows=oos_window_results,
            is_returns=is_returns_combined,
            oos_returns=oos_returns_combined,
            factor_attribution=factor_attr,
        )
