# regression/

Panel-OLS estimation of the volatility predictive regressions reported in the
paper. The module covers panel construction, the baseline and extension
specifications, and the volatility-, attention-, and sentiment-model
robustness checks.

## Layout

```
regression/
├── README.md
├── src/
│   ├── build_panel.py          CLI: build the regression panel
│   ├── run_regression.py       CLI: run all regressions + write summary tables
│   ├── config.py               Config dataclasses, paths, robustness lists
│   ├── merging/                Panel construction
│   │   ├── builder.py            Orchestrates `build_panel()`
│   │   ├── features.py           Attention, sentiment, lag transforms
│   │   ├── io.py                 Load / save the parquet panel
│   │   └── loaders.py            Load financial + sentiment inputs
│   ├── regression/             Estimation
│   │   ├── estimation.py         Estimation backend wrapper
│   │   ├── runner.py             Loops over horizons + robustness specs
│   │   └── specs.py              Spec factories (baseline, extension)
│   └── output/
│       └── writer.py             JSON / CSV / LaTeX writers
└── out/                        (gitignored) Output artefacts
```

## Empirical specification

Indexing: $i$ stock, $t$ trading day. Forecast horizon $h \in \{1, 3, 5\}$.

**Baseline:**

$$\ln \bar V^{(h)}_{i,t} = \alpha_i + \delta_t
  + \gamma_1 \tilde A_{i,t-h}
  + \gamma_2 \tilde S^{+}_{i,t-h} + \gamma_3 \tilde S^{-}_{i,t-h}
  + \gamma_4 (\tilde S^{+}_{i,t-h} \times \tilde A_{i,t-h})
  + \gamma_5 (\tilde S^{-}_{i,t-h} \times \tilde A_{i,t-h})
  + \gamma_6\, \mathit{ValidSent}_{i,t-h}
  + \gamma_7\, \ln \sigma^2_{i,t-h}
  + u_{i,t}$$

**Extension:** replaces the $S^{+}/S^{-}$ split by aggregate sentiment
$\tilde S$ and adds the magnitude $|\tilde S|$ and dispersion $\tilde D$
regressors.

- $\ln \bar V^{(h)}_{i,t}$ is the log root-mean-square of the daily variance
  series over the *backward* window $[t-h+1,\,t]$.
- The baseline daily variance is the jump-adjusted Garman–Klass (JA-GK)
  estimator (see [`financial_data/README.md`](../financial_data/README.md)).
- $\tilde A$ and $\tilde S$ denote within-stock z-scores of attention and
  sentiment.
- $\alpha_i$ stock fixed effects, $\delta_t$ date fixed effects.
- Standard errors clustered by stock.

## Inputs

- `financial_data/out/stock_returns.parquet` — daily returns + volatility
  measures.
- `financial_data/out/market_benchmark.csv` — S&P 500 benchmark.
- `reddit/out/stage3_aggregated/daily_sentiment.csv` — daily sentiment +
  mention counts (one variant per sentiment model).

If these are missing, run the upstream pipelines first:
`reddit/src/main.py`, `financial_data/src/fetch_stock_data.py`.

## Usage

```bash
cd regression/src

# Build the panel with the default configuration
python build_panel.py

# Force a full panel rebuild
python build_panel.py --force

# Run the baseline, extension, and robustness regressions
python run_regression.py

# Run only the primary specifications
python run_regression.py --skip-robustness

# List the planned runs without estimating
python run_regression.py --dry-run
```

> Panel-level descriptive statistics (summary tables, correlation matrix) are
> available as an optional analysis step in [`analysis/`](../analysis/README.md).

## Robustness checks

| Family               | Variants                                                            |
|----------------------|---------------------------------------------------------------------|
| Sentiment model      | `roberta_topic` (primary), `bertweet`, `fintwit`                    |
| Attention spec       | absolute / relative × global / within_stock                          |
| Volatility measure   | `ja_gk` (primary), `ja_rs`, `ja_pk`, `gk`, `pk`, `sqret`             |

Sentiment-model and attention-spec robustness are also run on the GARCH side
(see [`garch/README.md`](../garch/README.md)). Volatility-measure robustness
is panel-only.

## Outputs

```
out/
├── panel/
│   ├── panel_data.parquet
│   └── panel_metadata.json
└── regression/
    ├── raw/<group>/<horizon>/*.json     Full result objects
    └── tables/
        ├── summary_baseline.csv / .tex
        ├── summary_all.csv / .tex
        └── regression_results.tex       Standalone compilable LaTeX
```
