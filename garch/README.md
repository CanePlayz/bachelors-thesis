# garch/

Per-stock GARCH(1,1)-X estimation with social-media variables (attention,
positive and negative sentiment) entering the variance equation as exogenous
regressors, alongside controls for market volatility (VIX), the calendar
gap, and a sentiment-validity indicator. Population effects are aggregated
via the Mean Group (MG) estimator. Likelihood-ratio homogeneity tests on
the pooled-parameter panel-GARCH are run separately.

## Layout

```
garch/
├── README.md
├── src/
│   ├── run_garch.py        CLI: primary model + robustness checks
│   ├── run_lr_tests.py     CLI: LR tests for α, β, δ homogeneity
│   ├── config.py           Spec dataclass, default exogenous variables, paths
│   ├── data.py             Load the regression panel, slice into per-stock arrays
│   ├── model.py            Per-stock result container, Mean-Group aggregation, LaTeX/JSON writers
│   └── model_panel.py      Joint panel-GARCH MLE (production estimator)
└── out/                   (gitignored) Output artefacts
```

## Model

For stock $i$ on trading day $t$, with mean equation $r_{i,t} = \mu_i + \varepsilon_{i,t}$ ($\mu_i$ fixed at $0$):

$$\sigma^2_{i,t} = \omega_i + \alpha_i\,\varepsilon^2_{i,t-1}
                  + \beta_i\,\sigma^2_{i,t-1}
                  + \boldsymbol{\delta}_i' \, \mathbf{x}_{i,t-1}
                  + \gamma_i\,\widetilde{\text{VIX}}_{t-1}
                  + \varphi_i\,g_{t-1}
                  + \psi_i\,\text{ValidSent}_{i,t-1}$$

where $\mathbf{x}_{i,t-1} = (\tilde{A}_{i,t-1},\,\tilde{S}^{+}_{i,t-1},\,\tilde{S}^{-}_{i,t-1})'$
collects the standardized social-media variables (attention, positive and
negative sentiment), and the additional controls are:

- $\widetilde{\text{VIX}}_{t-1}$ — standardized CBOE VIX, market-volatility
  proxy / common factor.
- $g_{t-1}$ — calendar-gap variable controlling for mention-count inflation
  on pre-weekend / pre-holiday days (substitutes for the date fixed effects
  that the per-stock GARCH cannot include).
- $\text{ValidSent}_{i,t-1}$ — indicator distinguishing days with neutral
  posts from days with no posts at all.

Each stock has nine free parameters
$(\omega_i, \alpha_i, \beta_i, \delta_{i,1}, \delta_{i,2}, \delta_{i,3},
\gamma_i, \varphi_i, \psi_i)$.

Estimation is a single joint quasi-maximum likelihood problem solved by
`model_panel.estimate_panel_garch` (L-BFGS-B with block coordinate descent
over the global and stock-specific blocks). The pooling configuration is
controlled by the scopes on `GARCHSpec`; see *Default specification* below.
The cross-sectional average of the per-stock $\boldsymbol{\delta}_i$
estimates extracted from the joint solution is the Mean Group estimator
(Pesaran 1995), with the non-parametric Mean Group standard errors. A
sample-size-weighted variant is reported alongside.

## Default specification

The default `GARCHSpec` in `config.py` reflects the parameter-pooling
configuration selected by the likelihood-ratio battery in `run_lr_tests.py`:
$\alpha$ globally pooled (LR did not reject homogeneity at any conventional
significance level), $\beta$ and $\boldsymbol{\delta}$ stock-specific (LR
rejects pooling). The auxiliary controls $\gamma_i$, $\varphi_i$, $\psi_i$
are stock-specific throughout. `run_garch.py` calls the joint panel
estimator with these scopes and then extracts per-stock parameters from
the joint solution; `run_lr_tests.py` uses the same estimator under
alternative pooling restrictions to compute the LR statistics. To re-run
the homogeneity tests, edit the `base_spec` block in
`run_lr_tests.py:main()`.

## Inputs

- `regression/out/panel/panel_data.parquet` — built by
  `regression/src/build_panel.py`.

If absent, run `python regression/src/build_panel.py` first.

## Usage

```bash
cd garch/src

# Primary model + paper robustness checks
python run_garch.py

# Primary only
python run_garch.py --skip-robustness

# Override the per-stock minimum sample size
python run_garch.py --min-obs 200

# Likelihood-ratio homogeneity tests (alpha, beta, delta)
python run_lr_tests.py
```

## Robustness checks

The two GARCH-side robustness families described in the paper:

1. **Alternative sentiment models:** `bertweet`, `fintwit`.
2. **Alternative attention specifications:** absolute / relative × global /
   within_stock.

Volatility-measure robustness is panel-only — see
[`regression/README.md`](../regression/README.md).

## Outputs

```
out/
└── garch/
    ├── <run-name>/
    │   ├── stock_garch_results.csv / .parquet     Per-stock parameter estimates
    │   ├── mean_group_results.json                MG aggregates and per-stock distribution
    │   └── mean_group_table.tex                   LaTeX MG estimates
    └── lr_tests/                                  Likelihood-ratio test outputs
```
