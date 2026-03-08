"""
main.py
-------
Entry point for the Regime-Aware Multi-Factor Equity Model.

Usage:
    python main.py

Pipeline:
    1. Fetch & preprocess data (2018-2024)
    2. Run walk-forward backtest
    3. Compute performance metrics
    4. Print results table
    5. Generate & save all charts
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure UTF-8 output is not forced -- use safe ASCII for Windows cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.data     import DataLoader
from src.backtest import RegimeFactorBacktester
from src.metrics  import (
    compute_metrics,
    is_oos_comparison,
    summarize_oos_windows,
    cagr, sharpe_ratio, max_drawdown,
)
from src.plots    import generate_all_plots


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_START = "2018-01-01"
DATA_END   = "2024-12-31"

SEPARATOR  = "=" * 70
THIN_SEP   = "-" * 70


# ---------------------------------------------------------------------------
# Console printing helpers (ASCII only, safe for Windows cp1252)
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_df(df: pd.DataFrame) -> None:
    """Print a DataFrame with clean formatting."""
    print(df.to_string())
    print()


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Print the main performance table with a visual border."""
    print_header("PERFORMANCE METRICS")
    print_df(metrics_df)


def print_oos_summary(oos_df: pd.DataFrame) -> None:
    print_header("WALK-FORWARD OOS WINDOW SUMMARY")
    print_df(oos_df)


def print_is_oos(comparison_df: pd.DataFrame) -> None:
    print_header("IN-SAMPLE vs OUT-OF-SAMPLE (OVERFITTING CHECK)")
    print_df(comparison_df)


def print_regime_summary(regime_series: pd.Series) -> None:
    from src.regime import REGIME_NAMES
    print_header("REGIME DISTRIBUTION")
    counts   = regime_series.value_counts().sort_index()
    total    = counts.sum()
    for reg_id, count in counts.items():
        name = REGIME_NAMES.get(int(reg_id), f"State {reg_id}")
        pct  = count / total * 100
        bar  = "#" * int(pct / 2)
        print(f"  {name:10s} : {count:5d} days ({pct:5.1f}%)  {bar}")
    print()


def print_top_holdings_summary(rebalance_records: list) -> None:
    from src.regime import REGIME_NAMES
    print_header("LAST 5 REBALANCES SNAPSHOT")
    for rec in rebalance_records[-5:]:
        regime_name = REGIME_NAMES.get(rec.regime, "?")
        fw_str = "[" + ", ".join(f"{w:.2f}" for w in rec.factor_weights) + "]"
        print(f"  {str(rec.date.date()):12s} | Regime: {regime_name:8s} | "
              f"Factor Weights (Mom/Qual/LVol/Val): {fw_str}")
        print(f"    Holdings: {', '.join(rec.tickers)}")
        print(f"    Turnover: {rec.turnover:.0%}  |  TC: {rec.transaction_cost*10000:.1f} bps")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("  Regime-Aware Multi-Factor Equity Model")
    print("  Walk-Forward Backtest  |  2018-2024")
    print(SEPARATOR)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print_header("STEP 1: DATA LOADING")
    loader = DataLoader(start=DATA_START, end=DATA_END)

    # ------------------------------------------------------------------
    # Step 2: Run backtest
    # ------------------------------------------------------------------
    print_header("STEP 2: WALK-FORWARD BACKTEST")
    backtester = RegimeFactorBacktester(loader)
    result     = backtester.run()

    if result.portfolio_returns.empty:
        print("[ERROR] Backtest produced no returns. Check data availability.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Compute metrics
    # ------------------------------------------------------------------
    print_header("STEP 3: COMPUTING METRICS")

    # Align benchmark
    bench_ret = loader.bench.reindex(result.portfolio_returns.index).fillna(0.0)

    metrics_df = compute_metrics(
        result.portfolio_returns,
        bench_ret,
        label="Regime-Factor Strategy",
    )

    oos_summary = summarize_oos_windows(result.oos_windows)

    is_oos_df = is_oos_comparison(
        result.is_returns  if result.is_returns  is not None else pd.Series(dtype=float),
        result.oos_returns if result.oos_returns is not None else pd.Series(dtype=float),
    )

    # ------------------------------------------------------------------
    # Step 4: Print results
    # ------------------------------------------------------------------
    print_metrics_table(metrics_df)
    print_oos_summary(oos_summary)
    print_is_oos(is_oos_df)
    print_regime_summary(result.regime_series)
    print_top_holdings_summary(result.rebalance_records)

    # Per-window summary
    print_header("OOS WINDOW CAGR + SHARPE DETAIL")
    for w in result.oos_windows:
        print(f"  Window {w['window']} | {w['test_start'].date()} to "
              f"{w['test_end'].date()} | OOS Return: {w['oos_return']*100:.2f}%")
    print()

    # Summary headline
    strat_cagr   = cagr(result.portfolio_returns) * 100
    strat_sharpe = sharpe_ratio(result.portfolio_returns)
    strat_mdd    = max_drawdown(result.portfolio_returns) * 100
    bench_cagr   = cagr(bench_ret) * 100
    bench_sharpe = sharpe_ratio(bench_ret)

    print(SEPARATOR)
    print("  HEADLINE RESULTS")
    print(THIN_SEP)
    print(f"  Strategy CAGR   : {strat_cagr:+.2f}%   (SPY: {bench_cagr:+.2f}%)")
    print(f"  Strategy Sharpe : {strat_sharpe:.3f}     (SPY: {bench_sharpe:.3f})")
    print(f"  Strategy Max DD : {strat_mdd:.2f}%")
    print(f"  OOS Windows     : {len(result.oos_windows)}")
    print(f"  Total Rebalances: {len(result.rebalance_records)}")
    print(SEPARATOR)

    # ------------------------------------------------------------------
    # Step 5: Generate charts
    # ------------------------------------------------------------------
    generate_all_plots(
        result     = result,
        bench_close = loader.bench_close,
        metrics_df  = metrics_df,
    )

    print("\nDone. All charts saved to results/")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
