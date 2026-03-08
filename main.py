"""
main.py
-------
FinBERT NLP Earnings Sentiment Alpha Model
==========================================
Orchestration script — runs the full pipeline end-to-end.

Pipeline stages:
  1. Data    -- Download price data, simulate earnings events.
  2. NLP     -- Load FinBERT, score all earnings texts.
  3. Events  -- Run event study (market model, CAR, BHAR).
  4. Signals -- Construct long/short signals, compute IC.
  5. Backtest-- Event-driven backtest with transaction costs.
  6. Metrics -- Performance analytics, significance tests.
  7. Plots   -- Generate all charts to output/.

Usage:
  python main.py

Requirements:
  pip install transformers torch yfinance pandas numpy matplotlib seaborn scipy scikit-learn

NOTE: FinBERT model weights (~440 MB) will be downloaded from HuggingFace
on first run. Subsequent runs use the local cache.

IMPORTANT: All earnings transcript data is SIMULATED (see data.py).
The FinBERT model and event study methodology are genuine. See README.md.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Add src/ to path
# ---------------------------------------------------------------------------

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data        import load_all_data, UNIVERSE, MARKET_TICKER
from nlp         import load_finbert, score_events, label_from_composite
from event_study import run_event_study, summarize_by_sentiment
from signals     import generate_signals, merge_signals_with_events, compute_ic, quintile_car_summary
from backtest    import run_backtest, backtest_summary
from metrics     import (
    compute_ic_timeseries, ic_summary,
    car_by_sentiment_bucket,
    quintile_returns_table,
    portfolio_risk_metrics,
    pnl_attribution,
    print_performance_summary,
)
from plots import generate_all_plots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE          = "cpu"      # Set to "cuda" if GPU is available
RANDOM_SEED     = 42
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 60)
    print("  FINBERT EARNINGS ALPHA MODEL")
    print("  NLP + Event Study + Backtest Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: Data
    # ------------------------------------------------------------------
    print("\n[STAGE 1/7] Loading price data and simulating earnings events ...")
    data = load_all_data(seed=RANDOM_SEED)
    prices  = data["prices"]
    returns = data["returns"]
    events  = data["events"]

    print(f"  Price data: {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"  Earnings events: {len(events)} total")

    # ------------------------------------------------------------------
    # Stage 2: NLP — FinBERT scoring
    # ------------------------------------------------------------------
    print("\n[STAGE 2/7] Loading FinBERT and scoring earnings texts ...")
    try:
        tokenizer, model = load_finbert(device=DEVICE)
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install dependencies with: pip install transformers torch")
        sys.exit(1)

    events_scored = score_events(events, tokenizer, model, device=DEVICE)

    # Add categorical sentiment label
    events_scored["sentiment_label"] = events_scored["composite_score"].apply(
        lambda s: label_from_composite(s, threshold=0.1)
    )

    # Save scored events
    events_path = os.path.join(OUTPUT_DIR, "events_scored.csv")
    events_scored.drop(columns=["text"]).to_csv(events_path, index=False)
    print(f"  Scored events saved to: {events_path}")

    # Print sentiment distribution
    label_counts = events_scored["sentiment_label"].value_counts()
    print(f"  Sentiment distribution: {label_counts.to_dict()}")
    print(f"  Mean composite score:   {events_scored['composite_score'].mean():.4f}")

    # ------------------------------------------------------------------
    # Stage 3: Event Study
    # ------------------------------------------------------------------
    print("\n[STAGE 3/7] Running event study (market model, CAR, BHAR) ...")
    event_study_results, ar_panel = run_event_study(
        events=events_scored,
        returns=returns,
        market_col=MARKET_TICKER,
        verbose=True,
    )

    # Merge CAR results into scored events
    events_scored = events_scored.merge(
        event_study_results[["ticker", "event_date", "car_m1_p5", "car_0_p5", "bhar_20", "beta", "valid"]],
        left_on=["ticker", "earnings_date"],
        right_on=["ticker", "event_date"],
        how="left",
    )

    # Save event study results
    es_path = os.path.join(OUTPUT_DIR, "event_study_results.csv")
    event_study_results.to_csv(es_path, index=False)
    print(f"  Event study results saved to: {es_path}")

    print("\n  Market model summary:")
    valid_es = event_study_results[event_study_results["valid"]]
    print(f"    Mean beta:      {valid_es['beta'].mean():.3f}")
    print(f"    Mean R-squared: {valid_es['r_squared'].mean():.3f}")
    print(f"    Valid events:   {len(valid_es)} / {len(event_study_results)}")

    # ------------------------------------------------------------------
    # Stage 4: Signal Construction
    # ------------------------------------------------------------------
    print("\n[STAGE 4/7] Constructing trading signals ...")
    signals_df = generate_signals(
        events_scored=events_scored,
        n_long=5,
        n_short=5,
        hold_days=5,
    )

    # Merge into master frame
    master_df = merge_signals_with_events(
        signals_df=signals_df,
        events_scored=events_scored,
        event_study_results=event_study_results,
    )

    # Add sentiment label for downstream use (may be missing after merge)
    if "sentiment_label" not in master_df.columns:
        master_df["sentiment_label"] = master_df["composite_score"].apply(
            lambda s: label_from_composite(s, threshold=0.1) if pd.notna(s) else "neutral"
        )

    # Compute full-sample IC
    ic_result = compute_ic(master_df)
    print(f"  Information Coefficient (Spearman): {ic_result['ic_spearman']:.4f}  "
          f"(p={ic_result['p_value']:.4f}, significant={ic_result['significant']})")

    # Save master frame
    master_path = os.path.join(OUTPUT_DIR, "master_signals.csv")
    master_df.to_csv(master_path, index=False)
    print(f"  Master signals saved to: {master_path}")

    signal_counts = master_df["signal"].value_counts().to_dict()
    print(f"  Signal counts: Long={signal_counts.get(1,0)}, Short={signal_counts.get(-1,0)}, Neutral={signal_counts.get(0,0)}")

    # ------------------------------------------------------------------
    # Stage 5: Backtest
    # ------------------------------------------------------------------
    print("\n[STAGE 5/7] Running event-driven backtest ...")
    trades_df, equity_df = run_backtest(
        master_df=master_df,
        prices=prices,
        verbose=True,
    )

    if len(trades_df) > 0:
        trades_path = os.path.join(OUTPUT_DIR, "trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"  Trades saved to: {trades_path}")

    if len(equity_df) > 0:
        equity_path = os.path.join(OUTPUT_DIR, "equity_curve.csv")
        equity_df.to_csv(equity_path, index=False)
        print(f"  Equity curve saved to: {equity_path}")

    # ------------------------------------------------------------------
    # Stage 6: Analytics & Statistical Tests
    # ------------------------------------------------------------------
    print("\n[STAGE 6/7] Computing performance analytics ...")

    # Portfolio stats
    bt_stats   = backtest_summary(trades_df, equity_df) if len(trades_df) > 0 else {}
    risk_stats = portfolio_risk_metrics(equity_df) if len(equity_df) > 0 else {}

    # IC time-series
    ic_df      = compute_ic_timeseries(master_df)
    ic_stats   = ic_summary(ic_df)

    # CAR by sentiment bucket
    car_table     = car_by_sentiment_bucket(master_df)

    # Quintile analysis
    q_return_table = quintile_returns_table(master_df)
    q_car_table    = quintile_car_summary(master_df)

    # CAR significance summary
    car_sig = summarize_by_sentiment(
        events_with_scores=events_scored,
        results_df=event_study_results,
        sentiment_col="sentiment_label",
        car_col="car_m1_p5",
    )

    # P&L attribution
    attribution = pnl_attribution(trades_df) if len(trades_df) > 0 else pd.DataFrame()

    # Print full summary
    print_performance_summary(
        backtest_stats=bt_stats,
        ic_stats=ic_stats,
        car_table=car_table,
        quintile_table=q_return_table,
    )

    print("\n--- CAR Significance by Sentiment (t-test H0: mean=0) ---")
    print(car_sig.to_string())

    print("\n--- Quintile CAR Analysis ---")
    print(q_car_table.to_string())

    if not attribution.empty:
        print("\n--- P&L Attribution ---")
        print(attribution.to_string())

    # ------------------------------------------------------------------
    # Stage 7: Plots
    # ------------------------------------------------------------------
    print("\n[STAGE 7/7] Generating charts ...")
    saved_paths = generate_all_plots(
        ar_panel=ar_panel,
        events_scored=events_scored,
        master_df=master_df,
        quintile_table=q_car_table,
        ic_df=ic_df,
        equity_df=equity_df,
        prices=prices,
        event_study_results=event_study_results,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Charts generated: {len(saved_paths)}")
    print("=" * 60)

    return {
        "events_scored":      events_scored,
        "event_study_results": event_study_results,
        "master_df":          master_df,
        "trades_df":          trades_df,
        "equity_df":          equity_df,
        "bt_stats":           bt_stats,
        "ic_stats":           ic_stats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = main()
