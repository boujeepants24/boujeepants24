# Quant Research Lab

Four quantitative research projects in one place, being rebuilt to a single standard: pinned data, no look-ahead, out-of-sample testing, realistic costs, and tests that would catch the mistakes that make backtests look better than reality.

> **Read this first: no result in this repository is currently validated.** Each project was written before an audit (November 2026) that found real problems: look-ahead bias, double-counted costs, survivorship bias, and one project whose data is synthetic by design. Each project's README lists its specific issues at the top. Nothing here is investment advice or evidence of a profitable strategy.

## Projects

| Project | Question it asks | Data | Status | Known issues (details in each README) |
|---|---|---|---|---|
| [crypto-funding-arb](projects/crypto-funding-arb) | Can a delta-neutral position collect perpetual-futures funding payments after costs? | Binance public API (fetched live, not pinned) | Prototype | Closing costs counted twice; Kelly sizing written but not used; parameters tested in-sample |
| [kalman-factor-neutral](projects/kalman-factor-neutral) | Does Kalman-filtered beta hedging keep a momentum/quality long-short book market-neutral? | Yahoo Finance daily prices, 2015–2024 | Prototype | Look-ahead on rebalance days; hand-picked universe of today's large caps (survivorship bias) |
| [regime-factor-model](projects/regime-factor-model) | Should factor weights change with market regime (hidden Markov model)? | Yahoo Finance equities and crypto | Prototype | Regimes decoded with future data (Viterbi over the test window); rebalance-day look-ahead; regime weights fixed, not optimised |
| [finbert-earnings-alpha](projects/finbert-earnings-alpha) | Does earnings-call sentiment (FinBERT) predict post-earnings returns? | Real prices, **synthetic** text | Methodology test only | Text is generated from the future return, so the signal is built in. Tests the pipeline, not the hypothesis |

## The standard every project is being rebuilt to

A result counts only when all of these hold, and the README shows the command that produced it.

1. **Pinned data.** Raw data is saved with a checksum and a fetch date; the backtest reads the saved copy, so anyone can reproduce it.
2. **No look-ahead.** Decisions at time *t* use only data available before *t*; trades execute at *t+1* prices. Regime and filter states are causal (filtered, never smoothed or Viterbi-decoded over the future).
3. **Out-of-sample.** Parameters are chosen on a training period and reported on a later test period they never saw, ideally walk-forward.
4. **Point-in-time universe.** The stocks or coins available on each date, including ones later delisted. Where that data isn't available, the bias is stated next to the result.
5. **Costs.** Fees, spread and slippage are modelled on every trade, with a sensitivity table at 0.5×, 1× and 2× costs.
6. **Benchmarks and risk.** Compared with buy-and-hold and a simple baseline; reported with drawdown, volatility, turnover and exposure, not just return.
7. **Tests that catch cheating.** Unit tests for the accounting, plus a shuffled-label test: if the signal still "works" on randomly shuffled returns, something is leaking.
8. **Honest labels.** Synthetic data, short samples and in-sample results are labelled where the number appears.

## Rebuild plan

| Order | Work | Why first |
|---|---|---|
| 1 | Shared toolkit: data caching with checksums, a backtest loop that executes at *t+1*, cost model, metrics, and the shuffled-label test | Every project needs it, and it fixes the look-ahead bugs in one place |
| 2 | crypto-funding-arb on the toolkit | Real data, clearest economics, smallest fix list |
| 3 | kalman-factor-neutral, with the universe bias stated or fixed | |
| 4 | regime-factor-model with causal (forward-filtered) regimes | |
| 5 | finbert-earnings-alpha on real text (SEC 8-K earnings releases from EDGAR, free) | Turns a methodology demo into a real test |

## Layout

```
projects/<name>/      each original project, with its full commit history
```

Each project still runs on its own (`cd projects/<name> && pip install -r requirements.txt && python main.py`). The shared toolkit will live in `lab/` once the rebuild starts.

## History

These projects were separate repositories, merged here with their full commit history. This is now the only maintained copy.
