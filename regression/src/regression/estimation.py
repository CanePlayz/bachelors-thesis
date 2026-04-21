"""Panel regression estimation using pyfixest and linearmodels.

This module implements panel OLS estimation with:
- Entity (stock) fixed effects
- Time (date) fixed effects
- Clustered standard errors (entity, time, or two-way)

Uses pyfixest for efficient two-way fixed effects (iterative demeaning),
falling back to linearmodels.PanelOLS for single-dimension FE.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pyfixest
from linearmodels.panel import PanelOLS
from pyfixest import feols as feols  # type: ignore[attr-defined]

from .specs import RegressionResult, RegressionSpec


def run_panel_regression(
    df: pd.DataFrame,
    spec: RegressionSpec,
    verbose: bool = True,
) -> RegressionResult:
    """Run a panel regression with fixed effects.

    Uses pyfixest for two-way FE (much more memory-efficient),
    linearmodels for single-dimension FE.

    Args:
        df: Panel DataFrame with columns for target, regressors, ticker, date
        spec: Regression specification
        verbose: Print progress messages

    Returns:
        RegressionResult with all estimation statistics
    """
    if verbose:
        print(f"  Running: {spec.name} (target={spec.target})")

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------

    # Check required columns
    required_cols = [spec.target, "ticker", "date"] + spec.regressors
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Create working copy with required columns
    cols_to_copy = ["ticker", "date", spec.target] + spec.regressors
    work_df = df[cols_to_copy].copy()

    # Ensure date is datetime type
    work_df["date"] = pd.to_datetime(work_df["date"])

    # Replace inf values with NaN then drop
    work_df = work_df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with any NaN in target or regressors
    work_df = work_df.dropna()

    if len(work_df) == 0:
        if verbose:
            print(f"    WARNING: No valid observations for {spec.name}")
        return _empty_result(spec)

    # -------------------------------------------------------------------------
    # Choose estimation backend
    # -------------------------------------------------------------------------

    entity_effects = spec.entity_effects
    time_effects = spec.time_effects
    use_twoway = entity_effects and time_effects

    # Use pyfixest for two-way FE (memory-efficient)
    if use_twoway:
        return _run_pyfixest(work_df, spec, verbose)
    else:
        return _run_linearmodels(work_df, spec, verbose)


def _run_pyfixest(
    work_df: pd.DataFrame,
    spec: RegressionSpec,
    verbose: bool,
) -> RegressionResult:
    """Run regression using pyfixest (efficient for high-dimensional FE).

    pyfixest uses iterative demeaning (Gauss-Seidel) which is much more
    memory-efficient than creating dummy variable matrices.
    """

    # Build formula: y ~ x1 + x2 + ... | fe1 + fe2
    regressors_str = " + ".join(spec.regressors)

    # Fixed effects specification
    fe_parts = []
    if spec.entity_effects:
        fe_parts.append("ticker")
    if spec.time_effects:
        fe_parts.append("date")

    if fe_parts:
        fe_str = " + ".join(fe_parts)
        formula = f"{spec.target} ~ {regressors_str} | {fe_str}"
    else:
        formula = f"{spec.target} ~ {regressors_str}"

    # Clustering specification
    if spec.cluster == "twoway":
        vcov = {"CRV1": "ticker+date"}
    elif spec.cluster == "time":
        vcov = {"CRV1": "date"}
    else:
        vcov = {"CRV1": "ticker"}

    try:
        fitted = feols(formula, data=work_df, vcov=vcov)  # type: ignore[operator]
    except Exception as e:
        if verbose:
            print(f"    ERROR in pyfixest estimation: {e}")
        return _empty_result(spec, error=str(e))

    # -------------------------------------------------------------------------
    # Extract results
    # -------------------------------------------------------------------------

    result = RegressionResult(spec)  # type: ignore[call-arg]

    # Get coefficient table (index is variable name, not a column)
    coef_df = fitted.tidy()

    for var in spec.regressors:
        if var in coef_df.index:
            row = coef_df.loc[var]
            result.coefficients[var] = float(row["Estimate"])  # type: ignore[arg-type]
            result.std_errors[var] = float(row["Std. Error"])  # type: ignore[arg-type]
            result.t_stats[var] = float(row["t value"])  # type: ignore[arg-type]
            result.p_values[var] = float(row["Pr(>|t|)"])  # type: ignore[arg-type]

            # Confidence intervals (already in tidy output)
            result.conf_int_lower[var] = float(row["2.5%"])  # type: ignore[arg-type]
            result.conf_int_upper[var] = float(row["97.5%"])  # type: ignore[arg-type]

    # Model statistics (pyfixest uses underscore-prefixed attributes)
    result.r_squared = float(fitted._r2)  # type: ignore[union-attr]
    result.r_squared_within = float(fitted._r2_within)  # type: ignore[union-attr]
    result.n_obs = int(fitted._N)  # type: ignore[union-attr]
    result.n_entities = work_df["ticker"].nunique()
    result.n_time = work_df["date"].nunique()

    # F-test (pyfixest stores F-stat and p-value)
    try:
        result.f_stat = float(fitted._f_statistic)  # type: ignore[union-attr]
        result.f_pvalue = float(fitted._f_statistic_pvalue)  # type: ignore[union-attr]
    except Exception:
        result.f_stat = 0.0
        result.f_pvalue = 1.0

    # Additional model info
    result.model_info = {
        "entity_effects": spec.entity_effects,
        "time_effects": spec.time_effects,
        "cluster": spec.cluster,
        "backend": "pyfixest",
    }

    if verbose:
        n_sig = sum(1 for p in result.p_values.values() if p < 0.05)
        print(
            f"    → n={result.n_obs:,}, R²={result.r_squared:.4f}, "
            f"within-R²={result.r_squared_within:.4f}, {n_sig}/{len(spec.regressors)} sig."
        )

    return result


def _run_linearmodels(
    work_df: pd.DataFrame,
    spec: RegressionSpec,
    verbose: bool,
) -> RegressionResult:
    """Run regression using linearmodels.PanelOLS.

    Better for single-dimension FE; may run out of memory with two-way FE
    on large datasets.
    """

    entity_effects = spec.entity_effects
    time_effects = spec.time_effects

    # Set up MultiIndex for panel data (entity, time)
    work_df = work_df.set_index(["ticker", "date"])

    # Separate dependent and independent variables
    y = work_df[spec.target]
    X = work_df[spec.regressors]

    try:
        model = PanelOLS(
            dependent=y,
            exog=X,
            entity_effects=entity_effects,
            time_effects=time_effects,
            drop_absorbed=True,
            check_rank=False,
        )

        # Fit with clustered standard errors
        if spec.cluster == "twoway":
            fitted = model.fit(
                cov_type="clustered", cluster_entity=True, cluster_time=True
            )
        elif spec.cluster == "time":
            fitted = model.fit(cov_type="clustered", cluster_time=True)
        else:
            fitted = model.fit(cov_type="clustered", cluster_entity=True)

    except MemoryError:
        if verbose:
            print(
                f"    ERROR: Out of memory. Install pyfixest for efficient two-way FE:"
            )
            print(f"      pip install pyfixest")
        return _empty_result(spec, error="MemoryError: install pyfixest for two-way FE")

    except Exception as e:
        if verbose:
            print(f"    ERROR in estimation: {e}")
        return _empty_result(spec, error=str(e))

    # -------------------------------------------------------------------------
    # Extract results
    # -------------------------------------------------------------------------

    result = RegressionResult(spec)  # type: ignore[call-arg]

    # Coefficients and statistics
    for var in spec.regressors:
        if var in fitted.params.index:
            result.coefficients[var] = float(fitted.params[var])
            result.std_errors[var] = float(fitted.std_errors[var])
            result.t_stats[var] = float(fitted.tstats[var])
            result.p_values[var] = float(fitted.pvalues[var])

            # Confidence intervals
            ci = fitted.conf_int()
            if var in ci.index:
                lower_val: float = ci.loc[var, "lower"]  # type: ignore[assignment]
                upper_val: float = ci.loc[var, "upper"]  # type: ignore[assignment]
                result.conf_int_lower[var] = lower_val
                result.conf_int_upper[var] = upper_val

    # Model statistics
    result.r_squared = float(fitted.rsquared)
    result.r_squared_within = (
        float(fitted.rsquared_within)
        if hasattr(fitted, "rsquared_within")
        else result.r_squared
    )
    result.n_obs = int(fitted.nobs)
    result.n_entities = (
        int(fitted.entity_info["total"])
        if hasattr(fitted, "entity_info")
        else len(work_df.index.get_level_values(0).unique())
    )
    result.n_time = len(work_df.index.get_level_values(1).unique())

    # F-test for joint significance
    try:
        f_test = fitted.f_statistic
        result.f_stat = float(f_test.stat)
        result.f_pvalue = float(f_test.pval)
    except Exception:
        result.f_stat = 0.0
        result.f_pvalue = 1.0

    # Additional model info
    result.model_info = {
        "entity_effects": entity_effects,
        "time_effects": time_effects,
        "cluster": spec.cluster,
        "cov_type": getattr(fitted, "cov_type", "clustered"),
        "backend": "linearmodels",
    }

    if verbose:
        n_sig = sum(1 for p in result.p_values.values() if p < 0.05)
        print(
            f"    → n={result.n_obs:,}, R²={result.r_squared:.4f}, "
            f"within-R²={result.r_squared_within:.4f}, {n_sig}/{len(spec.regressors)} sig."
        )

    return result


def run_regression_spec(
    panel: pd.DataFrame,
    spec: RegressionSpec,
    verbose: bool = True,
) -> RegressionResult:
    """Convenience wrapper for run_panel_regression.

    Args:
        panel: Panel data DataFrame
        spec: Regression specification
        verbose: Print progress

    Returns:
        RegressionResult
    """
    return run_panel_regression(panel, spec, verbose=verbose)


def _empty_result(
    spec: RegressionSpec, error: Optional[str] = None
) -> RegressionResult:
    """Create an empty result for failed regressions."""
    result = RegressionResult(spec)  # type: ignore[call-arg]
    result.model_info["error"] = error or "No valid observations"
    return result


def generate_summary_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for panel variables.

    Args:
        panel: Panel DataFrame

    Returns:
        DataFrame with summary statistics (count, mean, std, min, max, etc.)
    """
    # Numeric columns only
    numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude index-like columns
    exclude = ["ticker", "date"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    stats = panel[numeric_cols].describe().T
    stats["n_missing"] = panel[numeric_cols].isna().sum()
    stats["pct_missing"] = stats["n_missing"] / len(panel) * 100

    return stats
