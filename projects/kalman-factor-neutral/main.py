"""
main.py
-------
Entry point for the Kalman Factor-Neutral strategy backtest.

Usage
-----
    python main.py

Output
------
- Prints performance metrics to stdout
- Saves 6 charts to results/
"""

import sys
import os

# Ensure the project root is on sys.path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data       import load_data, UNIVERSE, BENCHMARK, START_DATE, END_DATE
from src.kalman_beta import estimate_all_betas
from src.backtest   import run_backtest
from src.metrics    import compute_all_metrics, print_metrics, print_comparison_table
from src.plots      import save_all_plots


def main():
    print("=" * 60)
    print("  Kalman Factor-Neutral Strategy  |  Quant Portfolio")
    print("=" * 60)
    print(f"  Universe  : {len(UNIVERSE)} large-cap US stocks")
    print(f"  Period    : {START_DATE} to {END_DATE}")
    print(f"  Benchmark : {BENCHMARK}")
    print("=" * 60)
    print("")

    # -------------------------------------------------------------- #
    #  1. Load Data                                                   #
    # -------------------------------------------------------------- #
    returns, spy_returns = load_data()

    # -------------------------------------------------------------- #
    #  2. Kalman Beta Estimation (online, full history)               #
    # -------------------------------------------------------------- #
    beta_df, alpha_df = estimate_all_betas(returns, spy_returns)

    # -------------------------------------------------------------- #
    #  3. Run Backtest                                                #
    # -------------------------------------------------------------- #
    results = run_backtest(
        returns=returns,
        spy_returns=spy_returns,
        beta_df=beta_df,
        cost_bps=10.0,
        warmup=252,
        n_long=8,
        n_short=8,
    )

    # -------------------------------------------------------------- #
    #  4. Performance Metrics                                         #
    # -------------------------------------------------------------- #
    strategy_metrics = compute_all_metrics(
        results["daily_returns"],
        results["equity_curve"],
        spy_returns,
        label="Kalman Factor-Neutral",
    )

    # SPY metrics for comparison
    spy_aligned  = spy_returns.loc[results["daily_returns"].index]
    spy_equity_m = results["spy_equity"]

    spy_metrics = compute_all_metrics(
        spy_aligned,
        spy_equity_m,
        spy_aligned,
        label="SPY Buy-and-Hold",
    )

    print_metrics(strategy_metrics)
    print_comparison_table([strategy_metrics, spy_metrics])

    # -------------------------------------------------------------- #
    #  5. Rebalance Summary                                           #
    # -------------------------------------------------------------- #
    rlog = results["rebalance_log"]
    if rlog:
        avg_turnover = sum(r["turnover"] for r in rlog) / len(rlog)
        avg_cost     = sum(r["cost"]     for r in rlog) / len(rlog)
        print(f"  Rebalances   : {len(rlog)}")
        print(f"  Avg turnover : {avg_turnover*100:.1f}% per rebalance")
        print(f"  Avg cost     : {avg_cost*10000:.2f} bps per rebalance")
        print("")

    # -------------------------------------------------------------- #
    #  6. Save Charts                                                 #
    # -------------------------------------------------------------- #
    chart_paths = save_all_plots(results, beta_df)

    print("")
    print("All done.")
    print(f"Charts saved to: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')}")
    print("")


if __name__ == "__main__":
    main()
