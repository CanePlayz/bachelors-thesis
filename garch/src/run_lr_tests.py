"""Likelihood Ratio Tests for Panel-GARCH-X parameter pooling.

Tests whether specific GARCH parameters should be global (pooled across stocks)
or stock-specific, using the likelihood ratio test:

    LR = 2 * (LL_unrestricted - LL_restricted)
    LR ~ chi2(df)   where df = n_params_unrestricted - n_params_restricted

For each parameter (alpha, beta, delta), we compare:
    H0 (restricted):   parameter is global (shared across all N stocks)
    H1 (unrestricted): parameter is stock-specific (N separate values)
    df = N - 1  (going from 1 global to N stock-specific parameters)

Usage:
    python run_lr_tests.py                  # default: test all parameters
    python run_lr_tests.py --params alpha   # test only alpha
    python run_lr_tests.py --quick          # fewer BCD rounds for speed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from config import OUTPUT_DIR, GARCHSpec
from model_panel import PanelGARCHResult, ParamLayout, estimate_panel_garch
from scipy import stats

from data import StockData, load_panel

# =============================================================================
# Test design
# =============================================================================
#
# The pooling test on the exogenous block ("delta") is restricted to the
# *social-media* coefficients only. The remaining controls (VIX, calendar gap,
# valid-sentiment indicator) are treated as nuisance parameters: they enter the
# variance equation as stock-specific loadings in **both** the restricted and
# the unrestricted model, so they do not contribute to the LR statistic. This
# matches the symbol convention in the paper, where delta refers exclusively
# to the three social-media slopes.
#
# For the alpha and beta tests, the same nuisance treatment of the controls
# applies: in the base spec used as the shared restricted model, the
# social-media deltas are pooled while the controls are stock-specific.
CONTROL_COLS = ("vix_std", "gap_days", "valid_sentiment")


def social_media_cols(spec: GARCHSpec) -> List[str]:
    """Return the social-media subset of an exog-cols list (controls excluded)."""
    return [c for c in spec.exog_cols if c not in CONTROL_COLS]


def _build_restricted_spec(param: str, base_spec: GARCHSpec) -> GARCHSpec:
    """Build the restricted (H0) spec for the given LR test.

    Conventions:
    - alpha/beta tests: H0 has the parameter pooled (matches base).
    - delta test: H0 has the social-media deltas pooled and the controls
      stock-specific (matches base when base.delta_pooled_cols was set to
      social_media_cols(base_spec) by the runner).
    """
    spec_r = deepcopy(base_spec)
    if param == "delta":
        spec_r.delta_pooled_cols = social_media_cols(base_spec)
    else:
        setattr(spec_r, f"{param}_scope", "global")
    spec_r.verbose = False
    return spec_r


def _build_unrestricted_spec(param: str, base_spec: GARCHSpec) -> GARCHSpec:
    """Build the unrestricted (H1) spec for the given LR test.

    Conventions:
    - alpha/beta tests: H1 flips the parameter to stock-specific.
    - delta test: H1 makes the three social-media deltas stock-specific
      while keeping the controls stock-specific (delta_pooled_cols = []).
    """
    spec_u = deepcopy(base_spec)
    if param == "delta":
        spec_u.delta_pooled_cols = []
    else:
        setattr(spec_u, f"{param}_scope", "stock")
    spec_u.verbose = False
    return spec_u


def build_warm_start(
    restricted_result: PanelGARCHResult,
    unrestricted_spec: GARCHSpec,
    stocks: List[StockData],
    param_to_expand: str,
) -> np.ndarray:
    """Build initial theta for the unrestricted model from restricted estimates.

    Generic across both legacy global<->stock toggles and the per-column
    delta_pooled_cols scheme: each parameter is mapped slot-by-slot, replicating
    pooled values when the unrestricted model has stock-specific slots, and
    averaging stock-specific values when the unrestricted model is pooled.
    """
    r = restricted_result
    N = r.n_stocks
    K = len(unrestricted_spec.exog_cols)
    u_layout = ParamLayout(
        n_stocks=N,
        n_exog=K,
        spec=unrestricted_spec,
        exog_names=list(unrestricted_spec.exog_cols),
    )
    theta0 = np.zeros(u_layout.n_params)

    for pname in ["mu", "omega", "alpha", "beta"]:
        r_scope = getattr(r.spec, f"{pname}_scope")
        u_scope = getattr(unrestricted_spec, f"{pname}_scope")
        u_slice = u_layout.slices.get(pname)
        if u_slice is None:
            continue

        if pname == "mu" and r_scope == "zero" and u_scope == "zero":
            continue

        # Extract restricted value(s) via the layout getter
        if r_scope == "zero":
            val = 0.0
        elif r_scope == "global":
            val = getattr(r.layout, f"get_{pname}")(r.theta, 0)
        else:
            # stock-specific in restricted: keep per-stock values
            val = 0.0

        if u_scope == "global":
            if r_scope == "global" or r_scope == "zero":
                theta0[u_slice] = val
            else:
                # Stock to global (shouldn't happen in LR tests, but handle gracefully)
                vals = [getattr(r.layout, f"get_{pname}")(r.theta, i) for i in range(N)]
                theta0[u_slice] = np.mean(vals)
        elif u_scope == "stock":
            if r_scope == "global" or r_scope == "zero":
                # Replicate global value to all stocks
                theta0[u_slice] = val
            else:
                # Stock to stock: copy per-stock values
                for i in range(N):
                    theta0[u_slice.start + i] = getattr(r.layout, f"get_{pname}")(
                        r.theta, i
                    )

    # Delta: per-column transfer between the restricted and unrestricted layouts.
    if (
        u_layout.slices.get("delta") is not None
        and r.layout.slices.get("delta") is not None
    ):
        for k_u, col in enumerate(unrestricted_spec.exog_cols):
            if col not in r.spec.exog_cols:
                continue
            k_r = r.spec.exog_cols.index(col)
            r_slice = r.layout.delta_col_slices[k_r]
            r_pooled = r.layout.delta_col_pooled[k_r]
            u_slice = u_layout.delta_col_slices[k_u]
            u_pooled = u_layout.delta_col_pooled[k_u]
            r_vals = r.theta[r_slice]

            if u_pooled and r_pooled:
                theta0[u_slice.start] = float(r_vals[0])
            elif u_pooled and not r_pooled:
                theta0[u_slice.start] = float(np.mean(r_vals))
            elif (not u_pooled) and r_pooled:
                # Replicate single value to all N slots
                theta0[u_slice] = float(r_vals[0])
            else:
                # Both stock-specific: copy directly
                theta0[u_slice] = r_vals

    return theta0


@dataclass
class LRTestResult:
    """Result of a single likelihood ratio test."""

    param_name: str
    description: str
    ll_restricted: float  # LL under H0 (global)
    ll_unrestricted: float  # LL under H1 (stock-specific)
    n_params_restricted: int
    n_params_unrestricted: int
    df: int  # degrees of freedom = difference in params
    lr_stat: float  # 2 * (LL_unr - LL_res)
    p_value: float
    reject_h0: bool  # True = stock-specific is better
    aic_restricted: float
    aic_unrestricted: float
    bic_restricted: float
    bic_unrestricted: float
    elapsed_sec: float

    def summary_line(self) -> str:
        stars = ""
        if self.p_value < 0.001:
            stars = "***"
        elif self.p_value < 0.01:
            stars = "**"
        elif self.p_value < 0.05:
            stars = "*"
        verdict = "stock-specific" if self.reject_h0 else "GLOBAL"
        aic_winner = (
            "stock" if self.aic_unrestricted < self.aic_restricted else "global"
        )
        bic_winner = (
            "stock" if self.bic_unrestricted < self.bic_restricted else "global"
        )
        return (
            f"  {self.param_name:<12s} | "
            f"LR = {self.lr_stat:>12,.2f} | "
            f"df = {self.df:>5d} | "
            f"p = {self.p_value:.4e} {stars:<3s} | "
            f"AIC: {aic_winner:<6s} | "
            f"BIC: {bic_winner:<6s} | "
            f"=> {verdict}"
        )


def run_lr_test(
    param: str,
    stocks: List[StockData],
    base_spec: GARCHSpec,
    bcd_rounds: int = 60,
) -> LRTestResult:
    """Run LR test for a single parameter.

    Tests H0 (param is global) vs H1 (param is stock-specific).

    Parameters
    ----------
    param : str
        Which parameter to test: "alpha", "beta", "delta", "omega"
    stocks : list of StockData
        Panel data.
    base_spec : GARCHSpec
        Base specification (should have the tested param as global).
    bcd_rounds : int
        Maximum BCD rounds per model.

    Returns
    -------
    LRTestResult
    """
    N = len(stocks)

    # --- Restricted model (H0) ---
    spec_r = _build_restricted_spec(param, base_spec)

    print(f"  [{param}] Estimating restricted model (H0)...", flush=True)
    t0 = time.time()
    res_r = estimate_panel_garch(stocks, spec_r, skip_se=True)
    t_r = time.time() - t0
    print(
        f"  [{param}] Restricted:   LL = {res_r.loglik:>14,.2f} | "
        f"k = {res_r.n_params:>5d} | {t_r:.1f}s",
        flush=True,
    )

    # --- Unrestricted model (H1) ---
    spec_u = _build_unrestricted_spec(param, base_spec)

    print(f"  [{param}] Estimating unrestricted model (H1)...", flush=True)
    t0 = time.time()
    # Warm-start from restricted estimates
    theta0_warm = build_warm_start(res_r, spec_u, stocks, param)
    res_u = estimate_panel_garch(stocks, spec_u, theta0=theta0_warm, skip_se=True)
    t_u = time.time() - t0
    print(
        f"  [{param}] Unrestricted: LL = {res_u.loglik:>14,.2f} | "
        f"k = {res_u.n_params:>5d} | {t_u:.1f}s",
        flush=True,
    )

    # --- Compute LR test ---
    df = res_u.n_params - res_r.n_params
    lr_stat = 2 * (res_u.loglik - res_r.loglik)

    # LR stat must be non-negative in theory; if negative, models didn't converge well
    if lr_stat < 0:
        print(
            f"  [{param}] WARNING: negative LR stat ({lr_stat:.2f}) — "
            f"convergence issue, treating as 0",
            flush=True,
        )
        lr_stat = 0.0

    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df)) if df > 0 else 1.0
    reject_h0 = bool(p_value < 0.05)

    elapsed = t_r + t_u

    return LRTestResult(
        param_name=param,
        description=_describe_test(param),
        ll_restricted=res_r.loglik,
        ll_unrestricted=res_u.loglik,
        n_params_restricted=res_r.n_params,
        n_params_unrestricted=res_u.n_params,
        df=df,
        lr_stat=lr_stat,
        p_value=p_value,
        reject_h0=reject_h0,
        aic_restricted=res_r.aic,
        aic_unrestricted=res_u.aic,
        bic_restricted=res_r.bic,
        bic_unrestricted=res_u.bic,
        elapsed_sec=elapsed,
    )


def _describe_test(param: str) -> str:
    if param == "delta":
        return (
            "H0: social-media delta pooled (controls stock-specific) "
            "vs H1: all delta stock-specific"
        )
    return f"H0: {param} global vs H1: {param} stock-specific"


def run_all_lr_tests(
    stocks: List[StockData],
    base_spec: GARCHSpec,
    params: Optional[List[str]] = None,
    bcd_rounds: int = 60,
) -> List[LRTestResult]:
    """Run LR tests for multiple parameters.

    Caches the base model estimation since all restricted models are the same
    when the base spec has all tested params as global.

    Parameters
    ----------
    stocks : list of StockData
    base_spec : GARCHSpec
        Base specification with default scopes (typically all global except omega).
    params : list of str, optional
        Which parameters to test. Default: ["alpha", "beta", "delta"].
    bcd_rounds : int
        Maximum BCD rounds.

    Returns
    -------
    list of LRTestResult
    """
    if params is None:
        params = ["alpha", "beta", "delta"]

    # Normalize the base spec so the cached restricted model matches every test.
    # Convention: alpha=global, beta=global, social-media delta pooled,
    # control delta stock-specific.
    base_spec = deepcopy(base_spec)
    if base_spec.delta_pooled_cols is None:
        base_spec.delta_pooled_cols = social_media_cols(base_spec)

    # Pre-estimate the base model once — it serves as the restricted model
    # for every test in the new design (alpha and beta are global in base;
    # social-media delta is pooled in base; controls are stock-specific in both).
    base_result = None
    base_scope_matches = {param: True for param in params}

    t_base = 0.0
    needs_base = any(base_scope_matches.values())
    if needs_base:
        print(f"\n{'='*60}", flush=True)
        print(f"  Estimating base model (shared restricted model)", flush=True)
        print(f"{'='*60}", flush=True)
        spec_base = deepcopy(base_spec)
        spec_base.verbose = False
        t0 = time.time()
        base_result = estimate_panel_garch(stocks, spec_base, skip_se=True)
        t_base = time.time() - t0
        print(
            f"  Base model:    LL = {base_result.loglik:>14,.2f} | "
            f"k = {base_result.n_params:>5d} | {t_base:.1f}s\n",
            flush=True,
        )

    results = []
    for param in params:
        print(f"\n{'─'*60}", flush=True)
        print(f"  LR Test: {param}", flush=True)
        print(f"{'─'*60}", flush=True)

        if base_scope_matches[param] and base_result is not None:
            # Reuse cached base model as restricted
            result = _run_lr_test_with_cached_restricted(
                param, stocks, base_spec, base_result, t_base
            )
        else:
            result = run_lr_test(param, stocks, base_spec, bcd_rounds)

        results.append(result)
        print(f"  {result.summary_line()}", flush=True)

    return results


def _run_lr_test_with_cached_restricted(
    param: str,
    stocks: List[StockData],
    base_spec: GARCHSpec,
    res_r: PanelGARCHResult,
    t_r: float,
) -> LRTestResult:
    """Run LR test using a pre-estimated restricted model."""
    print(
        f"  [{param}] Restricted (cached): LL = {res_r.loglik:>14,.2f} | "
        f"k = {res_r.n_params:>5d}",
        flush=True,
    )

    # --- Unrestricted model (H1) ---
    spec_u = _build_unrestricted_spec(param, base_spec)

    print(f"  [{param}] Estimating unrestricted model (H1)...", flush=True)
    t0 = time.time()
    # Warm-start from restricted estimates
    theta0_warm = build_warm_start(res_r, spec_u, stocks, param)
    res_u = estimate_panel_garch(stocks, spec_u, theta0=theta0_warm, skip_se=True)
    t_u = time.time() - t0
    print(
        f"  [{param}] Unrestricted: LL = {res_u.loglik:>14,.2f} | "
        f"k = {res_u.n_params:>5d} | {t_u:.1f}s",
        flush=True,
    )

    # --- Compute LR test ---
    df = res_u.n_params - res_r.n_params
    lr_stat = 2 * (res_u.loglik - res_r.loglik)

    if lr_stat < 0:
        print(
            f"  [{param}] WARNING: negative LR stat ({lr_stat:.2f}) -- "
            f"convergence issue, treating as 0",
            flush=True,
        )
        lr_stat = 0.0

    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df)) if df > 0 else 1.0
    reject_h0 = bool(p_value < 0.05)

    elapsed = t_r + t_u

    return LRTestResult(
        param_name=param,
        description=_describe_test(param),
        ll_restricted=res_r.loglik,
        ll_unrestricted=res_u.loglik,
        n_params_restricted=res_r.n_params,
        n_params_unrestricted=res_u.n_params,
        df=df,
        lr_stat=lr_stat,
        p_value=p_value,
        reject_h0=reject_h0,
        aic_restricted=res_r.aic,
        aic_unrestricted=res_u.aic,
        bic_restricted=res_r.bic,
        bic_unrestricted=res_u.bic,
        elapsed_sec=elapsed,
    )


def print_lr_summary(results: List[LRTestResult]) -> None:
    """Print a summary table of all LR test results."""
    print(f"\n{'='*90}")
    print("  LIKELIHOOD RATIO TEST RESULTS: Global vs Stock-Specific Parameters")
    print(f"{'='*90}")
    print(
        f"  {'Parameter':<12s} | "
        f"{'LR stat':>12s} | "
        f"{'df':>5s} | "
        f"{'p-value':>12s}     | "
        f"{'AIC':>6s} | "
        f"{'BIC':>6s} | "
        f"{'Recommendation'}"
    )
    print(f"  {'─'*87}")

    for r in results:
        print(f"  {r.summary_line()}")

    print(f"{'='*90}")

    # Overall recommendation
    print("\n  Recommended specification:")
    for r in results:
        scope = "stock-specific" if r.reject_h0 else "global"
        ic_note = ""
        aic_prefers = "stock" if r.aic_unrestricted < r.aic_restricted else "global"
        bic_prefers = "stock" if r.bic_unrestricted < r.bic_restricted else "global"
        if aic_prefers != bic_prefers:
            ic_note = f" (AIC: {aic_prefers}, BIC: {bic_prefers})"
        elif (
            aic_prefers
            != ("stock-specific" if r.reject_h0 else "global").split("-")[0][:5]
        ):
            ic_note = f" (IC disagree: both prefer {aic_prefers})"
        print(f"    {r.param_name:<12s} -> {scope}{ic_note}")

    print()


def save_lr_results(results: List[LRTestResult], out_dir: Path) -> None:
    """Save LR test results to JSON and CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    data = []
    for r in results:
        data.append(
            {
                "param": r.param_name,
                "description": r.description,
                "ll_restricted": r.ll_restricted,
                "ll_unrestricted": r.ll_unrestricted,
                "n_params_restricted": r.n_params_restricted,
                "n_params_unrestricted": r.n_params_unrestricted,
                "df": r.df,
                "lr_stat": r.lr_stat,
                "p_value": r.p_value,
                "reject_h0": bool(r.reject_h0),
                "aic_restricted": r.aic_restricted,
                "aic_unrestricted": r.aic_unrestricted,
                "bic_restricted": r.bic_restricted,
                "bic_unrestricted": r.bic_unrestricted,
                "elapsed_sec": r.elapsed_sec,
            }
        )
    with open(out_dir / "lr_test_results.json", "w") as f:
        json.dump(data, f, indent=2)

    # CSV summary
    import pandas as pd

    df = pd.DataFrame(data)
    df.to_csv(out_dir / "lr_test_results.csv", index=False)

    print(f"  Results saved to {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LR Tests for Panel-GARCH-X Pooling")
    p.add_argument(
        "--params",
        nargs="+",
        choices=["alpha", "beta", "delta", "omega"],
        default=["alpha", "beta", "delta"],
        help="Parameters to test (default: alpha beta delta)",
    )
    p.add_argument("--min-obs", type=int, default=100)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fewer BCD rounds for faster (less precise) results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bcd_rounds = 30 if args.quick else 60

    # Base specification for the LR battery: the most-restricted model we
    # would consider --- alpha, beta, and the social-media delta all globally
    # pooled (controls remain stock-specific; see CONTROL_COLS). Each test
    # then flips one parameter to stock-specific and compares. We construct
    # this explicitly rather than calling default_spec() because the latter
    # now reflects the post-test selected configuration.
    base_spec = GARCHSpec(
        alpha_scope="global",
        beta_scope="global",
        delta_scope="global",
    )

    print("Loading panel data...", flush=True)
    stocks, _ = load_panel(base_spec, min_obs=args.min_obs)
    print(
        f"  {len(stocks)} stocks, {sum(s.n_obs for s in stocks):,} total obs",
        flush=True,
    )

    results = run_all_lr_tests(
        stocks, base_spec, params=args.params, bcd_rounds=bcd_rounds
    )
    print_lr_summary(results)
    save_lr_results(results, OUTPUT_DIR / "lr_tests")


if __name__ == "__main__":
    main()
