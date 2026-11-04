# FinBERT NLP Earnings Sentiment Alpha

> **Status: synthetic methodology test, not evidence of alpha.** The earnings text is generated from each event's *actual* post-earnings return, so sentiment is correlated with future returns by construction. Any IC, t-statistic or backtest result here measures that data generator, not FinBERT's ability to predict markets. The FinBERT pipeline and event-study maths are real; a test on real filings (e.g. SEC 8-K earnings releases) is planned.


An event-driven quantitative trading strategy that applies **FinBERT** — a BERT model pre-trained on financial text — to earnings announcements to generate sentiment signals and trade on post-announcement price drift.

---

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| Universe | 30 large-cap US stocks |
| Period | 2021-01-01 to 2024-12-31 |
| Signal Source | FinBERT composite score (positive_prob - negative_prob) |
| Entry | Close of earnings announcement day |
| Long Book | Top 5 positive sentiment stocks at each earnings batch |
| Short Book | Bottom 5 negative sentiment stocks at each earnings batch |
| Hold Period | 5 trading days |
| Transaction Cost | 10 bps per trade (one-way) |
| Position Sizing | Equal-weight within long and short books |

---

## IMPORTANT: Data Simulation Notice

> **The earnings transcript text data in this project is entirely simulated.**

Live earnings transcript APIs (Refinitiv, Bloomberg, Seeking Alpha) are paid services costing hundreds to thousands of dollars per month. To demonstrate the methodology without incurring these costs, this project:

1. Uses **actual price data** downloaded via `yfinance` (free, real market data).
2. **Simulates quarterly earnings announcement dates** aligned with typical reporting seasons.
3. **Generates synthetic earnings texts** using pools of realistic financial language sentences. The sentiment of the synthetic text is intentionally correlated with the actual post-announcement stock return (with realistic additive noise), so that FinBERT has a genuine signal to detect.

**What is real and methodologically correct:**
- The FinBERT model (`ProsusAI/finbert`) is the genuine published model.
- The NLP pipeline (chunking, averaging, composite scoring) is production-ready.
- The event study methodology (market model OLS, AR, CAR, BHAR, t-statistics) follows MacKinlay (1997).
- The backtest engine, IC analysis, and performance metrics are all genuine.

**The simulation is clearly documented** so this project can be used as a methodology demonstration and extended with real transcript data when API access is available.

---

## Project Structure

```
finbert-earnings-alpha/
├── src/
│   ├── data.py          # Price download + earnings event simulation
│   ├── nlp.py           # FinBERT pipeline: chunking, batch scoring, composite score
│   ├── event_study.py   # Market model OLS, AR/CAR/BHAR computation
│   ├── signals.py       # Signal construction, IC, quintile analysis
│   ├── backtest.py      # Event-driven backtester with transaction costs
│   ├── metrics.py       # IC time-series, risk metrics, CAR significance tests
│   └── plots.py         # 7 chart types: event study, IC scatter, equity curve, etc.
├── output/              # Generated charts (PNG) + CSVs (created on run)
├── main.py              # Full pipeline orchestration
├── requirements.txt
└── README.md
```

---

## NLP Pipeline: FinBERT

**Model:** `ProsusAI/finbert` ([HuggingFace](https://huggingface.co/ProsusAI/finbert))

FinBERT is a BERT model fine-tuned on ~10,000 financial news sentences from the Financial PhraseBank dataset. It outputs three class probabilities: `positive`, `negative`, `neutral`.

### Composite Score

```
composite_score = positive_prob - negative_prob
```

Range: [-1, +1]. Higher values indicate more bullish sentiment.

### Long-text Handling

Earnings transcripts can be thousands of words — far beyond BERT's 512-token limit. This pipeline:

1. Tokenizes the full text without special tokens.
2. Splits into chunks of 480 tokens.
3. Scores each chunk independently.
4. Returns a **weighted average** of probabilities across chunks (weighted by token count).

---

## Event Study Methodology

Following MacKinlay (1997):

### 1. Market Model Estimation

Estimated via OLS on a **120-day window ending 10 days before the event** (to avoid contamination):

```
R_i,t = alpha_i + beta_i * R_m,t + epsilon_i,t
```

Where `R_m` is the SPY (S&P 500 ETF) log return.

### 2. Abnormal Returns

```
AR_i,t = R_i,t - (alpha_i + beta_i * R_m,t)
```

The expected return from the market model is removed, isolating the firm-specific price reaction.

### 3. Cumulative Abnormal Return

```
CAR_i(tau1, tau2) = sum_{t=tau1}^{tau2} AR_i,t
```

Primary measure: **CAR(-1, +5)** — captures announcement-day and 5-day drift.

### 4. BHAR

```
BHAR_i = prod(1 + R_i,t) - prod(1 + E[R_i,t])  for t in [0, 19]
```

The geometric version of abnormal return, more appropriate for 20-day horizons.

### 5. Statistical Testing

Two-sided t-test: H0: mean(CAR) = 0.

Reported separately for positive, neutral, and negative sentiment groups.

---

## Outputs

After running `python main.py`, the `output/` directory contains:

| File | Description |
|------|-------------|
| `01_event_study_car.png` | Average CAR(-5, +20) by sentiment with 95% confidence bands |
| `02_ic_scatter.png` | FinBERT score vs 5-day realized return with OLS fit |
| `03_quintile_car.png` | Mean CAR(-1,+5) by sentiment quintile (monotonicity check) |
| `04_ic_timeseries.png` | Monthly IC time-series with 3-month rolling mean |
| `05_equity_curve.png` | Strategy NAV vs SPY buy-and-hold + drawdown subplot |
| `06_car_distribution.png` | Histogram of CAR distributions by sentiment label |
| `07_beta_distribution.png` | Distribution of estimated market betas |
| `events_scored.csv` | All earnings events with FinBERT scores |
| `event_study_results.csv` | CAR, BHAR, market model params per event |
| `master_signals.csv` | Combined signals + scores + CARs |
| `trades.csv` | Individual trade P&L records |
| `equity_curve.csv` | Daily portfolio NAV time-series |

---

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (FinBERT model downloaded on first run, ~440 MB)
python main.py
```

First run downloads FinBERT from HuggingFace Hub (~440 MB, cached for subsequent runs).

### GPU Acceleration

Change `DEVICE = "cpu"` to `DEVICE = "cuda"` in `main.py` if a CUDA GPU is available. This significantly speeds up the FinBERT inference stage.

---

## References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.
- MacKinlay, A.C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13-39.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
- ProsusAI/finbert model: https://huggingface.co/ProsusAI/finbert

---

## Extending to Real Data

To replace simulated transcripts with real data:

1. Obtain earnings transcript data from a provider (e.g., Seeking Alpha API, Refinitiv, Motley Fool).
2. Replace `simulate_earnings_events()` in `src/data.py` with a function that loads real transcripts.
3. The NLP pipeline in `src/nlp.py` is already designed to handle long texts of any length.
4. All downstream code (event study, signals, backtest) is unchanged.

The `forward_5d_return` column in the events DataFrame is used only for validation/IC analysis — the FinBERT signal itself is derived purely from the text.

---

*This is a portfolio demonstration project. All investment content is for educational purposes only.*
