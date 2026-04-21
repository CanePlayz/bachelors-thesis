# financial_data/

Daily stock market data from Yahoo Finance and the daily volatility battery
used by the panel and GARCH regressions.

## Layout

```
financial_data/
├── README.md
├── src/                       Source code
│   ├── fetch_stock_data.py      CLI entry point (Stages 1 & 2)
│   ├── config.py                Paths, dates, trading-day constants
│   ├── fetcher.py               Yahoo Finance download
│   ├── tickers.py               Ticker list selection / normalisation
│   ├── returns.py               Returns + daily volatility battery
│   └── output.py                Parquet/CSV writers, log files
└── out/                        (gitignored) Output artefacts
```

## Two stages

| Stage | Description                                                    | Output                                                |
|-------|----------------------------------------------------------------|-------------------------------------------------------|
| 1     | Fetch raw OHLCV from Yahoo Finance                             | `stock_prices.parquet`, `market_benchmark.csv`        |
| 2     | Compute daily returns and the daily volatility battery         | `stock_returns.parquet`                               |

Stages auto-discover and skip up-to-date outputs; staleness is detected via
file mtimes (Stage 2 is stale if `stock_prices.parquet` is newer than
`stock_returns.parquet`).

## Daily volatility measures

The **baseline** is the *jump-adjusted Garman–Klass* (JA-GK) estimator: the
squared overnight log return plus the Garman–Klass intraday range estimator.

- Overnight component: $\sigma^2_{\text{overnight},t} = (\ln(O_t / C_{t-1}))^2$
- Intraday Garman–Klass: $\sigma^2_{GK,t} = 0.5\,(\ln(H_t/L_t))^2 - (2\ln 2 - 1)(\ln(C_t/O_t))^2$

$$\sigma^2_{JA\text{-}GK,t} = \sigma^2_{\text{overnight},t} + \sigma^2_{GK,t}$$

Five **robustness alternatives** are also computed and stored alongside the
baseline:

| Column                | Definition                                                    |
|-----------------------|---------------------------------------------------------------|
| `volatility_ja_gk`    | Baseline: jump-adjusted Garman–Klass                           |
| `volatility_ja_rs`    | Jump-adjusted Rogers–Satchell                                  |
| `volatility_ja_pk`    | Jump-adjusted Parkinson                                        |
| `volatility_gk`       | Pure Garman–Klass (no overnight component)                     |
| `volatility_pk`       | Pure Parkinson (no overnight component)                        |
| `volatility_sqret`    | Squared close-to-close log return                              |

## Usage

The fetcher always sources its ticker universe and its date range from the
Reddit pipeline outputs (run that pipeline first).

Specifically:

- Tickers come from `reddit/out/stats/global/aggregated/ticker_stats.csv`
  (every ticker mentioned at least once, ETFs excluded).
- Start / end dates come from `reddit/out/stats/global/aggregated/daily_totals.csv`.

```bash
cd financial_data/src

# Auto-discover: run only the missing/stale stage(s)
python fetch_stock_data.py

# Force re-fetch everything
python fetch_stock_data.py --force

# Re-run only Stage 2 (recompute volatility from existing prices)
python fetch_stock_data.py --stage 2 --force

# Override the auto-discovered ticker list (debugging only).
# The date range is still taken from the Reddit output.
python fetch_stock_data.py --tickers AAPL MSFT TSLA GME --force

# All options
python fetch_stock_data.py --help
```

## Output schema

### `stock_prices.parquet`

| Column | Description |
|--------|-------------|
| `date` | Trading date |
| `ticker` | Stock symbol |
| `open`, `high`, `low`, `close` | OHLC prices |
| `adj_close` | Adjusted close |
| `volume` | Trading volume |

### `stock_returns.parquet`

| Column | Description |
|--------|-------------|
| `date`, `ticker` | Index columns |
| `open`, `high`, `low`, `close`, `adj_close` | Daily prices (kept for downstream use) |
| `simple_return` | Daily simple return |
| `log_return` | Daily log return |
| `volatility_*` | The six daily volatility series listed above |

### `market_benchmark.csv`

S&P 500 (`^GSPC`) close, daily returns, and the JA-GK baseline volatility.
