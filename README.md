# reddit-attention-volatility

Code accompanying the Bachelor's thesis *"The Role of Social Media Attention
and Sentiment in Short-Term Stock Volatility: Evidence from Reddit Data"*
(Jacob Pfundstein, University of Kaiserslautern-Landau, 2026).

The repository implements the full empirical pipeline from raw Reddit dumps
and Yahoo Finance prices to the panel-OLS and per-stock GARCH(1,1)-X
regressions reported in the paper.

## Repository layout

```
reddit-attention-volatility/
├── reddit/            Reddit ingestion, ticker matching, sentiment scoring
├── financial_data/    Stock prices and daily volatility measures
├── regression/        Panel construction and Panel OLS estimation
├── garch/             Per-stock GARCH(1,1)-X with Mean Group aggregation
└── analysis/          Optional analyses over pipeline outputs (e.g. descriptive stats)
```

Each module follows the same layout:

```
<module>/
├── README.md     Module documentation
├── src/          Source code (importable Python package)
└── out/          (gitignored) Output artefacts
```

## Methodology

The empirical specification follows the paper:

- **Panel OLS** with stock and date fixed effects and stock-clustered standard
  errors. Dependent variable is the log root-mean-square of the rolling daily
  volatility series. Forecast horizons $h \in \{1, 3, 5\}$ trading days.
- **Per-stock GARCH(1,1)-X** estimated by Gaussian quasi-maximum likelihood,
  with attention/sentiment entering the variance equation as exogenous
  regressors. Population effects are aggregated via the Mean Group estimator.
- **Daily volatility** is the jump-adjusted Garman–Klass (JA-GK) estimator
  (Molnár 2012): the squared overnight log return plus the Garman–Klass
  intraday range estimator.

Robustness checks (paper §6):

1. *Alternative sentiment models* — `bertweet`, `fintwit` (Panel + GARCH).
2. *Alternative attention specifications* — absolute / relative count and
   global / within-stock z-score (Panel + GARCH).
3. *Alternative volatility measures* — JA-RS, JA-PK, pure GK, pure PK,
   squared log returns (Panel only).

## Reproducing the results

```bash
# 0. Install dependencies (Python 3.13 recommended)
pip install -r requirements.txt

# 1. Ticker universe (optional: clean_tickers.csv is checked into the repo;
#    regenerate only when updating to the latest NASDAQ/NYSE listings)
python reddit/tickers/fetch_listings.py
python reddit/tickers/fetch_historical_tickers.py
python reddit/tickers/generate_clean_tickers.py

# 2. Reddit pipeline (requires r/wallstreetbets and r/options dumps placed
#    under reddit/datasets/, see reddit/README.md)
python reddit/src/main.py

# 3. Stock prices and daily volatility measures
python financial_data/src/fetch_stock_data.py

# 4. Build the regression panel
python regression/src/build_panel.py

# 5. Panel OLS regressions (baseline, extension, robustness)
python regression/src/run_regression.py
python analysis/src/descriptive_stats.py     # optional: panel summary tables

# 6. Per-stock GARCH(1,1)-X with Mean Group aggregation
python garch/src/run_garch.py
python garch/src/run_lr_tests.py             # optional: likelihood-ratio homogeneity tests
```

Outputs land under each module's `out/` directory (gitignored).

## Data sources

- **Reddit:** Pushshift dumps of `r/wallstreetbets` and `r/options`,
  distributed via the torrent file in `reddit/torrent/`. Extract the `.zst`
  archives into `reddit/datasets/` (or set `REDDIT_DATASETS_DIR` in the
  environment).
- **Stock prices:** Yahoo Finance.
- **Ticker universe:** NASDAQ and NYSE listings, see
  `reddit/tickers/clean_tickers.csv`.

## Disclaimer about AI usage

The codebase was developed with the assistance of GitHub Copilot, using the Claude Opus models. All code was reviewed and tested by the author.

## License

MIT — see [LICENSE](LICENSE).
