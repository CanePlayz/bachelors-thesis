"""Per-stock result container and Mean Group aggregation.

Estimation is done jointly across stocks by
`model_panel.estimate_panel_garch`; this module holds the per-stock
result dataclass and the aggregation utilities used downstream.

Model for stock i:
    r_{i,t} = eps_{i,t}               (zero-mean)
    sigma2_{i,t} = omega + alpha * eps2_{i,t-1} + beta * sigma2_{i,t-1}
                 + delta' * x_{i,t-1}

References:
  - Pesaran (1995): "Estimating long-run relationships from
    dynamic heterogeneous panels", Journal of Econometrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from config import GARCHSpec

# =============================================================================
# Per-Stock Result
# =============================================================================


@dataclass
class StockResult:
    """Estimation result for a single stock's GARCH(1,1)-X."""

    ticker: str
    n_obs: int
    converged: bool
    loglik: float
    aic: float
    bic: float

    omega: float
    alpha: float
    beta: float
    delta: Dict[str, float]  # exog_name -> coefficient
    delta_se: Dict[str, float]  # exog_name -> standard error

    persistence: float  # alpha + beta


# Per-stock MLE previously lived here; estimation is now done jointly
# across stocks by `model_panel.estimate_panel_garch`, and `run_garch.py`
# converts that panel result into a list of `StockResult` instances
# (one per stock) to feed the aggregation utilities below.


# =============================================================================
# Results DataFrame
# =============================================================================


def results_to_dataframe(results: List[StockResult]) -> pd.DataFrame:
    """Convert per-stock results to a tidy DataFrame."""
    rows = []
    for r in results:
        row = {
            "ticker": r.ticker,
            "n_obs": r.n_obs,
            "converged": r.converged,
            "loglik": r.loglik,
            "aic": r.aic,
            "bic": r.bic,
            "omega": r.omega,
            "alpha": r.alpha,
            "beta": r.beta,
            "persistence": r.persistence,
        }
        for name, val in r.delta.items():
            row[f"delta_{name}"] = val
        for name, val in r.delta_se.items():
            row[f"se_delta_{name}"] = val
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


# =============================================================================
# Mean Group Aggregation
# =============================================================================


@dataclass
class MeanGroupResult:
    """Aggregated delta estimates from the Mean Group estimator."""

    # Per exogenous variable
    exog_names: List[str]

    # Simple (unweighted) MG
    delta_mg: Dict[str, float]
    se_mg: Dict[str, float]
    t_mg: Dict[str, float]
    p_mg: Dict[str, float]

    n_stocks: int
    n_converged: int

    # Distribution of per-stock deltas
    delta_quantiles: Dict[str, Dict[str, float]]  # exog -> {q25, q50, q75, ...}
    delta_frac_positive: Dict[str, float]
    delta_frac_significant: Dict[str, float]  # |t| > 1.96 per stock


def mean_group_aggregate(
    results: List[StockResult],
    only_converged: bool = True,
) -> MeanGroupResult:
    """Compute the (equal-weighted) Mean Group estimator for delta across stocks.

    Follows Pesaran & Smith (1995): each stock contributes equally,
    standard errors are based on the cross-sectional variance:
        SE(delta_MG) = (1/sqrt(N)) * sqrt( (1/(N-1)) * sum( (delta_i - delta_MG)^2 ) )
    """
    from scipy import stats as sp_stats

    pool = [r for r in results if (r.converged or not only_converged)]
    if not pool:
        raise ValueError("No converged results to aggregate.")

    N = len(pool)
    exog_names = list(pool[0].delta.keys())

    # Collect per-stock arrays
    deltas: Dict[str, np.ndarray] = {}
    ses: Dict[str, np.ndarray] = {}

    for name in exog_names:
        d = np.array([r.delta[name] for r in pool])
        s = np.array([r.delta_se[name] for r in pool])
        deltas[name] = d
        ses[name] = s

    delta_mg, se_mg, t_mg, p_mg = {}, {}, {}, {}
    delta_quantiles = {}
    delta_frac_pos = {}
    delta_frac_sig = {}

    for name in exog_names:
        d = deltas[name]
        s = ses[name]

        # Filter out NaN/inf
        valid = np.isfinite(d)
        dv = d[valid]
        sv = s[valid]
        Nv = len(dv)

        if Nv < 2:
            for dct in [delta_mg, se_mg, t_mg, p_mg]:
                dct[name] = np.nan
            delta_quantiles[name] = {}
            delta_frac_pos[name] = np.nan
            delta_frac_sig[name] = np.nan
            continue

        # --- Simple MG ---
        mg = np.mean(dv)
        mg_se = np.std(dv, ddof=1) / np.sqrt(Nv)
        mg_t = mg / mg_se if mg_se > 0 else 0.0
        mg_p = 2 * (1 - sp_stats.t.cdf(abs(mg_t), df=Nv - 1))
        delta_mg[name] = mg
        se_mg[name] = mg_se
        t_mg[name] = mg_t
        p_mg[name] = mg_p

        # --- Distribution ---
        delta_quantiles[name] = {
            "min": float(np.min(dv)),
            "q05": float(np.percentile(dv, 5)),
            "q25": float(np.percentile(dv, 25)),
            "q50": float(np.median(dv)),
            "q75": float(np.percentile(dv, 75)),
            "q95": float(np.percentile(dv, 95)),
            "max": float(np.max(dv)),
            "std": float(np.std(dv, ddof=1)),
        }

        delta_frac_pos[name] = float(np.mean(dv > 0))

        # Fraction of stocks with individually significant delta
        valid_se = np.isfinite(sv) & (sv > 0)
        if np.sum(valid_se) > 0:
            t_individual = dv[valid_se] / sv[valid_se]
            delta_frac_sig[name] = float(np.mean(np.abs(t_individual) > 1.96))
        else:
            delta_frac_sig[name] = np.nan

    return MeanGroupResult(
        exog_names=exog_names,
        delta_mg=delta_mg,
        se_mg=se_mg,
        t_mg=t_mg,
        p_mg=p_mg,
        n_stocks=N,
        n_converged=sum(1 for r in pool if r.converged),
        delta_quantiles=delta_quantiles,
        delta_frac_positive=delta_frac_pos,
        delta_frac_significant=delta_frac_sig,
    )


# =============================================================================
# Printing / Output
# =============================================================================


def print_mean_group_table(mg: MeanGroupResult) -> None:
    """Pretty-print the Mean Group delta estimates."""

    def stars(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    print(f"\n{'='*90}")
    print("  MEAN GROUP ESTIMATES: Effect of Social Media on Volatility (delta)")
    print(f"  N = {mg.n_stocks} stocks ({mg.n_converged} converged)")
    print(f"{'='*90}")

    # Simple MG
    print(f"\n  Simple Mean Group (Pesaran, 1995):")
    print(f"  {'Variable':<28s} {'delta':>12s} {'SE':>12s} {'t':>8s} {'p':>10s}")
    print(f"  {'-'*74}")
    for name in mg.exog_names:
        print(
            f"  {name:<28s} "
            f"{mg.delta_mg[name]:>12.6f} "
            f"{mg.se_mg[name]:>12.6f} "
            f"{mg.t_mg[name]:>8.3f} "
            f"{mg.p_mg[name]:>10.4e} {stars(mg.p_mg[name])}"
        )

    # Distribution
    print(f"\n  Distribution of per-stock delta:")
    print(
        f"  {'Variable':<28s} {'median':>10s} {'IQR':>18s} "
        f"{'% > 0':>7s} {'% sig':>7s}"
    )
    print(f"  {'-'*74}")
    for name in mg.exog_names:
        q = mg.delta_quantiles[name]
        if not q:
            print(f"  {name:<28s} {'N/A':>10s} {'N/A':>18s} {'N/A':>7s} {'N/A':>7s}")
            continue
        print(
            f"  {name:<28s} "
            f"{q['q50']:>10.6f} "
            f"[{q['q25']:>8.6f}, {q['q75']:>8.6f}] "
            f"{mg.delta_frac_positive[name]:>6.1%} "
            f"{mg.delta_frac_significant[name]:>6.1%}"
        )

    print(f"\n{'='*90}")
