# Regime-Aware Multi-Factor Equity Model

A professional quantitative equity strategy that combines Hidden Markov Model (HMM) regime detection with a four-factor scoring model. The model dynamically adjusts factor weights based on the detected market regime and validates performance through rigorous walk-forward out-of-sample testing.

---

## Strategy Overview

### Universe
- **20 US Equities** (large-cap, multi-sector): AAPL, MSFT, NVDA, GOOGL, META, AMZN, JPM, GS, XOM, CVX, JNJ, UNH, TSLA, AMD, NFLX, COST, WMT, HD, V, MA
- **4 Crypto Assets** (via yfinance): BTC-USD, ETH-USD, SOL-USD, BNB-USD
- **Benchmark**: SPY

### Factors

| Factor | Signal | Window |
|--------|--------|--------|
| Momentum | 12-1 month cumulative log return (skip last month to avoid reversal) | 252 days |
| Quality | Trailing Sharpe ratio of daily returns (pure-price proxy) | 63 days |
| Low-Volatility | Negative of realized daily return volatility | 63 days |
| Value | Negative of price / 52-week high (distance from peak) | 252 days |

All factors are independently winsorized (5th-95th percentile) and cross-sectionally z-scored before combining.

### Regime Detection

A Gaussian HMM with 3 hidden states is trained on three features derived from SPY returns:
- 21-day rolling mean return (trend)
- 21-day rolling return volatility (risk)
- 63-day cumulative return (medium-term trend)

States are auto-labeled as **Bull**, **Bear**, or **High-Vol** by comparing emission distribution means.

### Regime-Conditional Factor Weights

| Regime | Momentum | Quality | Low-Vol | Value |
|--------|----------|---------|---------|-------|
| Bull | 40% | 25% | 15% | 20% |
| Bear | 10% | 35% | 35% | 20% |
| High-Vol | 15% | 20% | 50% | 15% |

### Portfolio Construction
- Monthly rebalancing on the last trading day of each month
- Long-only, top 8 assets by composite factor score
- Equal weight within selection (12.5% each)

### Transaction Costs
- Flat cost: 5 bps per round-trip trade
- Market impact: 10 bps per round-trip trade
- Applied proportional to one-way turnover

---

## Walk-Forward Validation

```
Train: 2 years (504 trading days)
Test:  6 months (126 trading days)
Step:  Roll forward by 1 test window
Min:   4 OOS windows required
```

The model is never tested on data it was trained on. HMM is re-fitted fresh for each walk-forward window.

---

## Project Structure

```
regime-factor-model/
├── src/
│   ├── __init__.py
│   ├── data.py          # yfinance fetching, alignment, return computation
│   ├── factors.py       # factor score computation + z-scoring
│   ├── regime.py        # HMM regime detection + weight optimization
│   ├── backtest.py      # walk-forward backtester with transaction costs
│   ├── metrics.py       # CAGR, Sharpe, Sortino, MDD, Calmar, Win Rate, etc.
│   └── plots.py         # all 8 visualizations
├── results/             # auto-created, charts saved here
├── main.py              # orchestrates full pipeline
├── requirements.txt
└── README.md
```

---

## Installation & Usage

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate.bat       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the backtest
python main.py
```

The console will print a full performance table. Charts are saved to `results/`.

---

## Output Charts

| File | Description |
|------|-------------|
| `equity_curves.png` | Cumulative returns: Strategy vs SPY with regime shading |
| `drawdowns.png` | Underwater equity chart (peak-to-trough drawdown) |
| `regime_overlay.png` | SPY price with HMM regime background colors |
| `factor_attribution.png` | Avg factor z-score of selected portfolio, by regime |
| `oos_vs_is.png` | Walk-forward OOS window returns + IS vs OOS comparison |
| `regime_distribution.png` | Time spent in each regime |
| `monthly_returns_heatmap.png` | Year x month return calendar heatmap |
| `rolling_sharpe.png` | 6-month rolling Sharpe: Strategy vs SPY |

---

## Performance Metrics Computed

- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio (annualized, Rf = 0)
- Sortino Ratio (downside deviation)
- Maximum Drawdown
- Calmar Ratio (CAGR / Max Drawdown)
- Win Rate (% of months with positive return)
- Hit Ratio (% of months outperforming SPY)
- Information Ratio (active return / tracking error)

---

## Design Decisions

**Why a pure-price Quality factor?**
yfinance does not expose ROE or book-value data reliably. A trailing Sharpe ratio computed from price returns is a recognized pure-price proxy for earnings quality that correlates with accounting Quality factors in the literature.

**Why blend optimized weights 50/50 with prior weights?**
Regularizing the data-driven regime weights prevents over-fitting to short historical regime windows, especially for regimes with few observations.

**Why equal weight within selection?**
Equal weighting avoids concentration risk and is known to outperform market-cap weighting in cross-sectional momentum strategies (DeMiguel et al. 2009).

**Why skip the most recent month in Momentum?**
Short-term return reversal (Jegadeesh 1990) is well-documented. Skipping the last ~21 trading days avoids buying recent losers that are short-term mean-reverting.

---

## Dependencies

```
yfinance >= 0.2.36
pandas   >= 2.0.0
numpy    >= 1.24.0
hmmlearn >= 0.3.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.10.0
statsmodels >= 0.14.0
```

---

## Disclaimer

This project is for educational and research purposes only. Nothing in this repository constitutes financial advice or a recommendation to buy or sell any security. Past backtest performance does not guarantee future results.
