"""
main.py
-------
Entry point for the Crypto Funding Rate Arbitrage Engine.

Pipeline
--------
1. Fetch data from Binance public API (funding rates + OHLCV)
2. Optionally cache fetched data to disk (parquet) for faster re-runs
3. Run the event-driven backtester
4. Compute performance metrics and print summary
5. Generate all charts to output/plots/

Usage
-----
  python main.py                    # full run (fetch live data)
  python main.py --cache            # use cached data if available
  python main.py --cache --no-plots # skip chart generation
  python main.py --help
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress benign matplotlib / pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Ensure src/ is on the path regardless of working directory
# ---------------------------------------------------------------------------
SRC_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data    import load_universe, ASSETS
from backtest import FundingArbBacktester
from metrics  import performance_summary, print_summary, monthly_pnl_attribution
from plots    import generate_all_plots

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(ROOT_DIR, "output", "cache")
PLOTS_DIR  = os.path.join(ROOT_DIR, "output", "plots")
CACHE_FILE = os.path.join(CACHE_DIR, "universe_data.pkl")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save_cache(data: dict) -> None:
    """Persist the fetched universe dict to disk using pickle."""
    import pickle
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Data cached to: {CACHE_FILE}")


def _load_cache() -> dict:
    """Load previously cached universe dict from disk."""
    import pickle
    print(f"Loading data from cache: {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto Funding Rate Arbitrage Backtester",
    )
    parser.add_argument(
        "--cache", action="store_true",
        help="Use cached data from a previous run if available; "
             "otherwise fetch and save to cache.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip chart generation (useful for fast performance testing).",
    )
    parser.add_argument(
        "--nav", type=float, default=1_000_000.0,
        help="Initial portfolio NAV in USD (default: 1,000,000).",
    )
    parser.add_argument(
        "--entry-threshold", type=float, default=0.0003,
        help="Minimum |funding rate| to open a position (decimal, default: 0.0003).",
    )
    parser.add_argument(
        "--exit-threshold", type=float, default=0.0001,
        help="Funding rate below which an open position is closed (default: 0.0001).",
    )
    parser.add_argument(
        "--basis-stop", type=float, default=1.0,
        help="Adverse basis move in %% that triggers an exit (default: 1.0).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every trade and funding payment during backtest.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("  CRYPTO FUNDING RATE ARBITRAGE ENGINE")
    print("=" * 60)
    print(f"  Initial NAV       : ${args.nav:,.0f}")
    print(f"  Entry threshold   : {args.entry_threshold*100:.4f}%  "
          f"(~{args.entry_threshold * 3 * 365 * 100:.0f}% ann.)")
    print(f"  Exit threshold    : {args.exit_threshold*100:.4f}%")
    print(f"  Basis stop        : {args.basis_stop:.1f}%")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    if args.cache and os.path.exists(CACHE_FILE):
        data = _load_cache()
    else:
        print("\n[1/4] Fetching data from Binance public API...")
        data = load_universe(ASSETS)
        if args.cache:
            _save_cache(data)

    funding_wide = data["funding"]
    perp_dict    = data["perp"]
    spot_dict    = data["spot"]

    print(f"      Funding rate data: {len(funding_wide)} rows x {len(funding_wide.columns)} assets")
    print(f"      Date range: {funding_wide.index[0].date()} to {funding_wide.index[-1].date()}")

    # ------------------------------------------------------------------
    # Step 2: Backtest
    # ------------------------------------------------------------------
    print("\n[2/4] Running event-driven backtest...")
    bt = FundingArbBacktester(
        funding_wide     = funding_wide,
        perp_dict        = perp_dict,
        spot_dict        = spot_dict,
        initial_nav      = args.nav,
        entry_threshold  = args.entry_threshold,
        exit_threshold   = args.exit_threshold,
        basis_stop_pct   = args.basis_stop,
        verbose          = args.verbose,
    )
    results = bt.run()

    ledger      = results["ledger"]
    trades_df   = results["trades"]
    payments_df = results["payments"]

    # ------------------------------------------------------------------
    # Step 3: Performance metrics
    # ------------------------------------------------------------------
    print("\n[3/4] Computing performance metrics...")
    summary = performance_summary(ledger, initial_nav=args.nav)
    print_summary(summary)

    # Monthly attribution table
    monthly = monthly_pnl_attribution(ledger)
    print("\n  Monthly P&L Attribution ($):")
    print(f"  {'Month':<10}  {'Funding':>12}  {'Basis':>12}  "
          f"{'Fees':>10}  {'Borrow':>10}  {'Net':>12}")
    print("  " + "-" * 62)
    for period, row in monthly.iterrows():
        print(
            f"  {str(period):<10}  "
            f"${row['funding_income']:>11,.0f}  "
            f"${row['basis_pnl']:>+11,.0f}  "
            f"${-row['fee_cost']:>9,.0f}  "
            f"${-row['borrow_cost']:>9,.0f}  "
            f"${row['net_pnl']:>+11,.0f}"
        )

    # Trade summary
    if not trades_df.empty:
        n_trades = len(trades_df)
        total_fees = trades_df["fee_usd"].sum() + trades_df["slippage_usd"].sum()
        print(f"\n  Total executions  : {n_trades}")
        print(f"  Total transaction cost: ${total_fees:,.2f}")
    if not payments_df.empty:
        n_payments   = len(payments_df)
        total_funding = payments_df["payment_usd"].sum()
        print(f"  Funding payments  : {n_payments}")
        print(f"  Total funding USD : ${total_funding:+,.2f}")

    # ------------------------------------------------------------------
    # Step 4: Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print(f"\n[4/4] Generating charts -> {PLOTS_DIR}")
        generate_all_plots(
            ledger       = ledger,
            funding_wide = funding_wide,
            perp_dict    = perp_dict,
            spot_dict    = spot_dict,
            initial_nav  = args.nav,
            output_dir   = PLOTS_DIR,
        )
    else:
        print("\n[4/4] Plot generation skipped (--no-plots).")

    print("\nDone. Results saved to output/")
    return results


if __name__ == "__main__":
    main()
