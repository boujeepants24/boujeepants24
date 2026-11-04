# Kalman Factor-Neutral Strategy

> **Status: research prototype, results not validated.** An audit (Nov 2026) found:
> - **Look-ahead bias:** on rebalance days the new weights are built from returns that include that day, and are then applied to that same day's return (`src/backtest.py`). Weights should take effect from the next day.
> - **Survivorship bias:** the 40-stock universe is a hand-picked list of today's large caps applied back to 2015.
>
> Performance numbers from this version should not be relied on. Fixes are tracked in the Quant Research Lab rebuild.


A quantitative long/short equity strategy combining momentum + quality factor
signals with time-varying Kalman filter beta hedging.

## Strategy Overview

- **Universe**: 40 large-cap US stocks across Technology, Financials, Energy, and Healthcare
- **Signal**: Composite factor score = 0.5 * momentum (12-1 month) + 0.5 * quality (rolling Sharpe)
- **Portfolio**: Long top 8 stocks, short bottom 8 stocks (equal weight per leg)
- **Market neutrality**: Kalman filter tracks each stock's time-varying beta vs SPY; residual net beta is hedged with a SPY overlay
- **Rebalancing**: Monthly (last trading day of each month)
- **Transaction costs**: 10 bps one-way (round-trip deducted at each rebalance)
- **Backtest period**: 2015-01-01 to 2024-12-31

## Why It Works

The strategy captures two well-documented equity risk premia:

1. **Momentum premium** (Jegadeesh & Titman, 1993): Past 12-1 month winners continue to outperform losers over the next month
2. **Quality premium** (Novy-Marx, 2013): Firms with consistent, high-quality returns trade at a discount relative to their fundamentals

By going long top-ranked stocks and short bottom-ranked stocks, the portfolio
captures both premia while remaining dollar-neutral. The Kalman beta hedge
eliminates residual market exposure, producing a portfolio with near-zero
correlation to SPY.

## Kalman Filter Beta Estimation

For each stock i, the state-space model is:

```
R_i(t) = alpha_i + beta_i(t) * R_SPY(t) + epsilon_i(t)

State:  x = [beta_i, alpha_i]
Design: H = [R_SPY,  1      ]

Process: x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
Obs:     y_t = H_t @ x_t + v_t, v_t ~ N(0, R)
```

The filter runs online (one update per trading day), allowing betas to evolve
smoothly across market cycles without look-ahead bias.

## Factor Definitions

| Factor   | Definition                                           |
|----------|------------------------------------------------------|
| Momentum | Cumulative log return, days [-252, -21]              |
| Quality  | Rolling 63-day annualised Sharpe ratio of returns    |

Both factors are cross-sectionally z-scored, then averaged with equal weight.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output

Charts saved to `results/`:

| File                    | Description                              |
|-------------------------|------------------------------------------|
| 01_equity_curve.png     | Strategy vs SPY vs 60/40                 |
| 02_drawdown.png         | Drawdown comparison                      |
| 03_monthly_heatmap.png  | Calendar monthly return heatmap          |
| 04_leg_attribution.png  | Long leg vs short leg cumulative returns |
| 05_rolling_sharpe.png   | Rolling 252-day Sharpe ratio             |
| 06_kalman_betas.png     | Portfolio beta evolution over time       |

## Code Structure

```
kalman-factor-neutral/
├── src/
│   ├── data.py         # yfinance price fetch, returns
│   ├── factors.py      # momentum + quality factor scores
│   ├── kalman_beta.py  # Kalman filter beta estimation (numpy only)
│   ├── portfolio.py    # long/short construction + beta neutralization
│   ├── backtest.py     # monthly rebalancing backtester with costs
│   ├── metrics.py      # CAGR, Sharpe, Sortino, Max DD, Calmar, correlation
│   └── plots.py        # 6 publication-quality charts
├── main.py
├── requirements.txt
└── README.md
```

## Academic References

- Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling Losers"
- Novy-Marx, R. (2013). "The Other Side of Value: The Gross Profitability Premium"
- Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
