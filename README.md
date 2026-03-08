# Crypto Funding Rate Arbitrage Engine

A professional-grade quantitative backtesting engine for the **perpetual futures funding rate arbitrage** strategy — one of the most capital-efficient strategies historically deployed by crypto hedge funds (Three Arrows Capital, Alameda Research, Wintermute, etc.).

---

## What Is Funding Rate Arbitrage?

### The Mechanism

Crypto perpetual futures do not have an expiry date. To keep their price anchored to the spot price, exchanges use a **funding rate** mechanism:

- Every 8 hours, a funding payment is exchanged between longs and shorts
- If `perp_price > spot_price` (perp at premium), the **funding rate is positive** — longs pay shorts
- If `perp_price < spot_price` (perp at discount), the **funding rate is negative** — shorts pay longs

The rate is computed as:

```
funding_rate = clamp(premium_index + clamp(interest_rate - premium_index, -0.05%, 0.05%), -0.75%, 0.75%)
```

Where `premium_index = (perp_mid - spot_mid) / spot_mid`

### The Trade

This strategy exploits persistent positive (or negative) funding by entering a **delta-neutral** position:

| Funding Condition         | Perp Leg    | Spot Leg   | P&L Source           |
|---------------------------|-------------|------------|----------------------|
| Positive (perp > spot)    | Short perp  | Long spot  | Collect funding paid by longs |
| Negative (perp < spot)    | Long perp   | Short spot | Collect funding paid by shorts |

Because the perp and spot positions offset each other, the strategy has **near-zero directional (delta) exposure to BTC/ETH/etc.** The P&L comes almost entirely from the funding payments.

### Why It Works

Funding rates exhibit strong **mean-reversion with persistence**: once funding is elevated, it tends to stay elevated for multiple 8h windows before reverting. This is driven by market structure — when sentiment is strongly bullish (retail buying perpetuals), longs pile in and funding stays positive for days. This persistence is what makes the edge predictable and backtestable.

**Annualized funding rate** = `funding_rate_per_8h * 3 * 365`

For example, a funding rate of `0.03%` per 8h annualizes to `~33% per year` — collected in a delta-neutral position.

---

## Architecture

```
crypto-funding-arb/
├── src/
│   ├── data.py          # Binance public API: funding rates + OHLCV (no key needed)
│   ├── signals.py       # Entry/exit signal logic, autocorrelation, regime detection
│   ├── sizing.py        # Kelly criterion position sizing with portfolio constraints
│   ├── execution.py     # Taker fees (4bps/leg), slippage model, funding payments
│   ├── risk.py          # Delta tracking, OU basis model, VaR/CVaR, liquidation guard
│   ├── backtest.py      # Event-driven backtester (8h interval processing)
│   ├── metrics.py       # Sharpe, Sortino, CAGR, max DD, P&L attribution
│   └── plots.py         # 10 professional charts (heatmaps, equity curve, ACF, etc.)
├── main.py              # Entry point with CLI
├── requirements.txt
└── README.md
```

---

## Data Sources

All data is fetched from **Binance public API** — no API key required.

| Data Type         | Endpoint                                      |
|-------------------|-----------------------------------------------|
| Funding rates     | `https://fapi.binance.com/fapi/v1/fundingRate` |
| Perp OHLCV        | `https://fapi.binance.com/fapi/v1/klines`      |
| Spot OHLCV        | `https://api.binance.com/api/v3/klines`        |

**Universe**: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, AVAXUSDT, LINKUSDT, ADAUSDT, DOGEUSDT

**History**: 2 years of 8-hour bars (~2,190 periods per asset)

---

## Signal Logic

### Entry

A position is opened when the absolute funding rate exceeds `entry_threshold`:

```python
entry_threshold = 0.0003   # 0.03% per 8h = ~33% annualized
```

- `funding_rate > 0.03%`  → SHORT perp, LONG spot  (collect from longs)
- `funding_rate < -0.03%` → LONG perp, SHORT spot  (collect from shorts)

### Exit Conditions

A position is closed when **any** of the following occur:

1. **Funding decay**: `|funding_rate| < exit_threshold` (0.01% per 8h)
2. **Basis stop**: the basis moves adversely by more than `basis_stop_pct` (1.0%) from entry
3. **Liquidation risk**: leverage approaches the 3x maximum (handled in `risk.py`)

---

## Position Sizing: Kelly Criterion

The optimal position size uses the Kelly formula adapted for a continuous payoff distribution:

```
f* = mu / sigma^2

where:
  mu    = funding_rate - fee_per_period - borrow_per_period   (net edge per 8h)
  sigma = rolling std of basis changes (OU diffusion parameter)
```

We use **half-Kelly** (`f* * 0.5`) for risk management conservatism.

Hard constraints override Kelly:
- Maximum 20% of NAV per asset
- Maximum 100% gross notional exposure
- 60% size reduction when 4+ assets have correlated high funding simultaneously

---

## Transaction Cost Model

All costs are realistically modeled:

| Cost Component         | Rate                       | Applied To                |
|------------------------|----------------------------|---------------------------|
| Taker fee              | 4 bps per leg              | All executions (open/close) |
| Slippage               | 2 bps per $1M notional     | Per leg (linear impact model) |
| Borrow cost            | 2% per year                | Spot leg notional          |

Full round-trip cost (open + close, both legs): **16 bps fees + slippage**

---

## Risk Management

### Delta Neutrality
Net delta exposure is tracked at every 8h step. The strategy is designed to maintain `delta ≈ 0`. Alerts trigger if `|net_delta| > 5% of NAV`.

### Basis Risk: Ornstein-Uhlenbeck Model
The basis spread `(perp - spot) / spot` is modelled as an OU mean-reverting process:

```
dX = kappa * (theta - X) * dt + sigma * dW
```

The fitted `kappa` (mean-reversion speed) and `sigma` (diffusion) are used to:
1. Estimate a 3-sigma stress scenario for basis adverse moves
2. Scale position size via the Kelly criterion (sigma enters the denominator)
3. Trigger risk rejection for high-sigma regimes

### VaR and CVaR
Historical simulation Value-at-Risk (95%) and Conditional VaR (Expected Shortfall) are computed on the 8h return series using a 90-period rolling window.

### Leverage Limit
- Hard cap: 3x gross notional / NAV
- Margin buffer: 20% of margin reserved at all times
- Pre-trade gate rejects or scales down trades that would breach these limits

---

## Performance Metrics

| Metric              | Description                                                   |
|---------------------|---------------------------------------------------------------|
| CAGR                | Compound annual growth rate over backtest period              |
| Sharpe Ratio        | Annualized: mean / std * sqrt(3 * 365)                        |
| Sortino Ratio       | Sharpe but only penalises downside volatility                 |
| Calmar Ratio        | CAGR / |Max Drawdown|                                         |
| Max Drawdown        | Largest peak-to-trough decline                                |
| Funding Yield (ann) | Total funding income / avg NAV, annualized                    |
| Win Rate            | % of 8h periods with positive P&L                            |
| Profit Factor       | Sum of gains / sum of losses                                  |
| Return Skewness     | Expected to be positive (many small gains, rare losses)       |

---

## Charts Generated

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `01_equity_curve.png`         | Strategy NAV vs BTC buy-and-hold (normalized)            |
| `02_funding_rate_heatmap.png` | Asset x time heatmap of annualized funding rates         |
| `03_monthly_attribution.png`  | Stacked bar: funding / basis / fees / borrow by month    |
| `04_return_distribution.png`  | Histogram + KDE of 8h returns vs normal distribution     |
| `05_drawdown.png`             | Underwater equity curve with max DD annotation           |
| `06_acf.png`                  | Funding rate autocorrelation by asset (confirms persistence)|
| `07_funding_timeseries.png`   | Per-asset annualized funding rate over time              |
| `08_basis_timeseries.png`     | Per-asset basis spread (perp premium) over time          |
| `09_monthly_funding_heatmap.png` | Asset x month funding yield heatmap                  |
| `10_leverage_positions.png`   | Leverage and position count over time                    |

---

## Installation and Usage

```bash
# Clone / navigate
cd crypto-funding-arb

# Install dependencies
pip install -r requirements.txt

# Run full backtest (fetches live data from Binance)
python main.py

# Cache data on first run, reuse on subsequent runs
python main.py --cache

# Skip chart generation for faster iteration
python main.py --cache --no-plots

# Custom parameters
python main.py --nav 500000 --entry-threshold 0.0004 --basis-stop 1.5

# Full help
python main.py --help
```

---

## Key Parameters

| Parameter           | Default | Description                                       |
|---------------------|---------|---------------------------------------------------|
| `--nav`             | 1000000 | Initial portfolio value (USD)                     |
| `--entry-threshold` | 0.0003  | Min funding rate to enter (decimal, per 8h)       |
| `--exit-threshold`  | 0.0001  | Funding rate at which position is closed          |
| `--basis-stop`      | 1.0     | Adverse basis move (%) that triggers exit         |
| `--cache`           | False   | Cache/reuse data from Binance                     |
| `--no-plots`        | False   | Skip chart generation                             |
| `--verbose`         | False   | Print individual trade events                     |

---

## Strategy Limitations and Risks

1. **Execution risk**: In live trading, leg execution is not simultaneous. A sudden price move between perp and spot fills creates residual delta.

2. **Liquidation risk**: In a violent market crash, the spot leg may lose value faster than the perp short profits — if exchange margin is insufficient, the perp position can be force-liquidated before the hedge pays off.

3. **Exchange counterparty risk**: Funds held on the exchange are at risk (cf. FTX collapse, Nov 2022).

4. **Funding rate manipulation**: Funding rates can spike and reverse quickly around major events. The basis stop loss addresses this but does not eliminate the risk.

5. **Crowding**: As this strategy has become widely known, elevated funding increasingly reflects the strategy's own demand for shorts rather than organic market structure.

6. **Negative funding streaks**: When the market is in a sustained bear regime, funding may stay negative for extended periods. The strategy handles this (long perp / short spot) but short selling spot adds borrow costs and liquidity risk.

---

## Background: Who Runs This Strategy

Funding rate arbitrage has been one of the most consistent alpha sources in crypto:

- **Alameda Research** reportedly ran this at scale, using cross-exchange basis trades
- **Three Arrows Capital** deployed leveraged versions during 2020-2021 bull runs
- **Market neutral hedge funds** (Elwood, Fasanara) run systematic versions today
- At peak bull market (Nov 2021), BTC funding rates averaged 0.1%+ per 8h (~110% annualized) for weeks — pure delta-neutral carry

The Sharpe ratio of a well-executed funding arb strategy has historically been 2-5x during high-funding regimes, making it one of the best risk-adjusted strategies available in crypto markets.

---

## Project Structure Notes

This project is structured as a professional quant library:

- **No hardcoded paths** — all paths are relative to the project root
- **No API keys required** — Binance public endpoints only
- **Rate limiting** — 0.25s sleep between requests, exponential retry on failure
- **Data caching** — optional pickle cache to avoid re-fetching on repeated runs
- **Clean separation of concerns**: data, signals, sizing, execution, risk, backtest, metrics, plots
- **Typed, docstrings throughout** — production-ready code style
