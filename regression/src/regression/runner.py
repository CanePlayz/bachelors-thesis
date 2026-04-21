"""Regression runner for orchestrating all regression analyses.

This module provides high-level functions to run:
- Baseline regressions (return and volatility for each horizon)
- Robustness checks (alternative sentiment models)
- Extension regressions (with dispersion and intensity)

Results are organized by regression type and saved to appropriate directories.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from config import (ATTENTION_ROBUSTNESS_SPECS, OUTPUT_DIR, AttentionConfig,
                    RegressionConfig)

from .estimation import run_panel_regression
from .specs import (RegressionResult, RegressionSpec,
                    create_baseline_volatility_spec,
                    create_extension_volatility_spec)


def run_all_regressions(
    panel: pd.DataFrame,
    config: RegressionConfig,
    output_dir: Path | None = None,
    skip_robustness: bool = False,
) -> Dict[str, List[RegressionResult]]:
    """Run all configured regressions.

    This is the main entry point for running the full regression analysis.
    Runs baseline, extension, and optionally robustness regressions for
    each horizon and target type.

    Args:
        panel: Panel DataFrame with all required variables
        config: Regression configuration
        output_dir: Output directory for results
        skip_robustness: Skip robustness checks

    Returns:
        Dictionary mapping regression type to list of results:
        {
            "baseline_return": [...],
            "baseline_volatility": [...],
            "extension_volatility": [...],
            "robustness_bertweet_return": [...],
            ...
        }
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "regression"

    all_results: Dict[str, List[RegressionResult]] = {}

    print("\n" + "=" * 60)
    print("RUNNING REGRESSION ANALYSIS")
    print("=" * 60)
    start_time = time.time()

    # -------------------------------------------------------------------------
    # Baseline volatility regressions
    # -------------------------------------------------------------------------
    print("\n[1/5] Running baseline volatility regressions...")

    baseline_results = run_baseline_regressions(panel, config)
    all_results.update(baseline_results)

    # -------------------------------------------------------------------------
    # Extension regressions (dispersion + intensity)
    # -------------------------------------------------------------------------
    print("\n[2/5] Running extension regressions...")

    extension_results = run_extension_regressions(panel, config)
    all_results.update(extension_results)

    # -------------------------------------------------------------------------
    # Robustness regressions (sentiment models + volatility measures + attention)
    # -------------------------------------------------------------------------
    if not skip_robustness:
        print("\n[3/5] Running robustness: alternative sentiment models...")
        robustness_sent = run_robustness_sentiment(panel, config)
        all_results.update(robustness_sent)

        print("\n[4/5] Running robustness: alternative volatility measures...")
        robustness_vol = run_robustness_volatility(panel, config)
        all_results.update(robustness_vol)

        print("\n[5/5] Running robustness: alternative attention specifications...")
        robustness_att = run_robustness_attention(panel, config)
        all_results.update(robustness_att)
    else:
        print("\n[3/5] Skipping robustness regressions")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    total_regressions = sum(len(v) for v in all_results.values())

    print("\n" + "=" * 60)
    print("REGRESSION ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total regressions: {total_regressions}")
    print(f"Total time: {elapsed:.1f}s")

    return all_results


def run_baseline_regressions(
    panel: pd.DataFrame,
    config: RegressionConfig,
) -> Dict[str, List[RegressionResult]]:
    """Run baseline volatility regressions for all horizons.

    Args:
        panel: Panel DataFrame
        config: Regression configuration

    Returns:
        Dictionary with "baseline_volatility" results
    """
    results = {
        "baseline_volatility": [],
    }

    # Determine fixed effects and clustering from config
    fe_type = config.estimation.fixed_effects
    entity_effects = fe_type in ("entity", "twoway")
    time_effects = fe_type in ("time", "twoway")
    cluster = config.estimation.cluster_se

    split_pos_neg = config.sentiment.split_pos_neg

    print(f"\n  Horizons: {config.horizons}")
    print(f"  Fixed effects: entity={entity_effects}, time={time_effects}")
    print(f"  Clustering: {cluster}")

    for h in config.horizons:
        print(f"\n  Horizon h={h}:")

        # Volatility regression
        vol_spec = create_baseline_volatility_spec(
            horizon=h,
            split_pos_neg=split_pos_neg,
            entity_effects=entity_effects,
            time_effects=time_effects,
            cluster=cluster,
        )
        vol_result = run_panel_regression(panel, vol_spec, verbose=True)
        results["baseline_volatility"].append(vol_result)

    return results


def run_extension_regressions(
    panel: pd.DataFrame,
    config: RegressionConfig,
) -> Dict[str, List[RegressionResult]]:
    """Run extended volatility regressions with dispersion and intensity.

    Args:
        panel: Panel DataFrame
        config: Regression configuration

    Returns:
        Dictionary with "extension_volatility" results
    """
    results = {
        "extension_volatility": [],
    }

    # Check if dispersion and intensity columns exist
    if (
        "sentiment_disp_std" not in panel.columns
        or "sentiment_intensity_std" not in panel.columns
    ):
        print(
            "  WARNING: Dispersion or intensity columns not found, skipping extensions"
        )
        return results

    fe_type = config.estimation.fixed_effects
    entity_effects = fe_type in ("entity", "twoway")
    time_effects = fe_type in ("time", "twoway")
    cluster = config.estimation.cluster_se

    for h in config.horizons:
        print(f"\n  Horizon h={h}:")

        ext_spec = create_extension_volatility_spec(
            horizon=h,
            entity_effects=entity_effects,
            time_effects=time_effects,
            cluster=cluster,
        )
        ext_result = run_panel_regression(panel, ext_spec, verbose=True)
        results["extension_volatility"].append(ext_result)

    return results


def run_robustness_sentiment(
    panel: pd.DataFrame,
    config: RegressionConfig,
) -> Dict[str, List[RegressionResult]]:
    """Run robustness regressions with alternative sentiment models.

    For each robustness model, loads the model-specific sentiment data,
    rebuilds the sentiment columns, and runs baseline VOLATILITY regressions only.

    Args:
        panel: Base panel DataFrame (will copy and modify sentiment columns)
        config: Regression configuration

    Returns:
        Dictionary with robustness results keyed by model
    """
    from merging.features import (compute_interaction_terms,
                                  compute_sentiment_variables, standardize)
    from merging.loaders import (load_sentiment_data,
                                 map_calendar_to_trading_day)

    results = {}

    print(f"  Robustness models: {config.robustness_sentiment_models}")

    for model in config.robustness_sentiment_models:
        key_vol = f"robustness_{model}_volatility"
        results[key_vol] = []

        print(f"\n  Model: {model}")
        print(f"    Loading sentiment data for {model}...")

        # Load sentiment data for this model
        sentiment_df = load_sentiment_data(config, model=model)

        # Get trading calendar from panel dates
        trading_calendar = pd.DataFrame({"date": sorted(panel["date"].unique())})

        # Map sentiment to trading days
        sentiment_df = map_calendar_to_trading_day(
            sentiment_df,
            trading_calendar,
            date_col="date",
        )

        # Aggregate sentiment to trading day level
        sentiment_df = _aggregate_sentiment_for_robustness(sentiment_df)

        # Create model-specific panel by joining sentiment to base panel
        model_panel = _create_model_panel(panel, sentiment_df, config)

        print(f"    Panel prepared: {len(model_panel):,} rows with valid sentiment")

        fe_type = config.estimation.fixed_effects
        entity_effects = fe_type in ("entity", "twoway")
        time_effects = fe_type in ("time", "twoway")
        cluster = config.estimation.cluster_se
        split_pos_neg = config.sentiment.split_pos_neg

        for h in config.horizons:
            print(f"\n    Horizon h={h}:")

            # Volatility regression only
            vol_spec = create_baseline_volatility_spec(
                horizon=h,
                split_pos_neg=split_pos_neg,
                entity_effects=entity_effects,
                time_effects=time_effects,
                cluster=cluster,
            )
            vol_spec.name = f"robustness_{model}_volatility_h{h}"
            vol_spec.description = f"Robustness ({model}): Volatility, h={h}"

            vol_result = run_panel_regression(model_panel, vol_spec, verbose=True)
            results[key_vol].append(vol_result)

    return results


def run_robustness_volatility(
    panel: pd.DataFrame,
    config: RegressionConfig,
) -> Dict[str, List[RegressionResult]]:
    """Run robustness regressions with alternative volatility measures.

    Re-runs baseline volatility regression using each of the five alternative
    volatility measures (jump-adjusted Rogers--Satchell, jump-adjusted Parkinson,
    pure Garman--Klass, pure Parkinson, and squared close-to-close returns)
    as the dependent variable (with matching lagged variance control).

    Args:
        panel: Panel DataFrame with all volatility columns pre-computed
        config: Regression configuration

    Returns:
        Dictionary with robustness results keyed by measure
    """
    results = {}

    fe_type = config.estimation.fixed_effects
    entity_effects = fe_type in ("entity", "twoway")
    time_effects = fe_type in ("time", "twoway")
    cluster = config.estimation.cluster_se
    split_pos_neg = config.sentiment.split_pos_neg

    for measure in config.volatility.robustness_measures:
        key = f"robustness_{measure}_volatility"
        results[key] = []

        print(f"\n  Volatility measure: {measure}")

        for h in config.horizons:
            print(f"    Horizon h={h}:")

            vol_spec = create_baseline_volatility_spec(
                horizon=h,
                split_pos_neg=split_pos_neg,
                entity_effects=entity_effects,
                time_effects=time_effects,
                cluster=cluster,
                vol_measure=measure,
            )
            vol_spec.name = f"robustness_{measure}_volatility_h{h}"
            vol_spec.description = f"Robustness ({measure}): Volatility, h={h}"

            vol_result = run_panel_regression(panel, vol_spec, verbose=True)
            results[key].append(vol_result)

    return results


def run_robustness_attention(
    panel: pd.DataFrame,
    config: RegressionConfig,
) -> Dict[str, List[RegressionResult]]:
    """Run robustness regressions with alternative attention specifications.

    Re-computes attention using each of the 3 alternative (measure, standardization)
    combinations and re-runs baseline volatility regressions.

    The four specifications are:
    - abs_within (primary, already in baseline — skipped here)
    - abs_global
    - rel_within
    - rel_global

    Args:
        panel: Panel DataFrame
        config: Regression configuration

    Returns:
        Dictionary with robustness results keyed by attention spec
    """
    from merging.features import (compute_attention_variables,
                                  compute_interaction_terms,
                                  compute_lagged_predictors)

    results = {}

    fe_type = config.estimation.fixed_effects
    entity_effects = fe_type in ("entity", "twoway")
    time_effects = fe_type in ("time", "twoway")
    cluster = config.estimation.cluster_se
    split_pos_neg = config.sentiment.split_pos_neg

    for att_config in ATTENTION_ROBUSTNESS_SPECS:
        spec_label = f"{att_config.measure}_{att_config.standardization}"
        key = f"robustness_att_{spec_label}_volatility"
        results[key] = []

        print(f"\n  Attention spec: {spec_label}")

        # Create modified config with this attention spec
        from dataclasses import replace as dc_replace

        alt_config = dc_replace(config, attention=att_config)

        # Re-compute attention on a copy of the panel
        alt_panel = panel.copy()

        # Drop old attention-dependent columns
        att_drop = [
            c
            for c in alt_panel.columns
            if any(
                c.startswith(prefix)
                for prefix in [
                    "attention",
                    "sentiment_pos_x_attention",
                    "sentiment_neg_x_attention",
                    "sentiment_x_attention",
                ]
            )
        ]
        alt_panel = alt_panel.drop(columns=att_drop, errors="ignore")

        # Re-compute attention
        alt_panel = compute_attention_variables(alt_panel, alt_config)

        # Re-compute interaction terms
        alt_panel = compute_interaction_terms(alt_panel, alt_config)

        # Re-compute lagged predictors
        alt_panel = compute_lagged_predictors(alt_panel, config.horizons)

        for h in config.horizons:
            print(f"    Horizon h={h}:")

            vol_spec = create_baseline_volatility_spec(
                horizon=h,
                split_pos_neg=split_pos_neg,
                entity_effects=entity_effects,
                time_effects=time_effects,
                cluster=cluster,
            )
            vol_spec.name = f"robustness_att_{spec_label}_volatility_h{h}"
            vol_spec.description = (
                f"Robustness (attention={spec_label}): Volatility, h={h}"
            )

            vol_result = run_panel_regression(alt_panel, vol_spec, verbose=True)
            results[key].append(vol_result)

    return results


def _aggregate_sentiment_for_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment data to trading day level for robustness checks.

    Uses weight_sum (engagement weights) for sentiment re-aggregation,
    consistent with the main panel builder.
    """
    agg_dict = {
        "mention_count": "sum",
    }

    if "sentiment_avg" in df.columns:
        df["_weighted_sentiment"] = df["sentiment_avg"] * df["weight_sum"]
        agg_dict["_weighted_sentiment"] = "sum"
        agg_dict["weight_sum"] = "sum"

    if "sentiment_disp" in df.columns:
        agg_dict["sentiment_disp"] = "max"

    result = df.groupby(["trading_date", "ticker"]).agg(agg_dict).reset_index()

    if "_weighted_sentiment" in result.columns:
        result["sentiment_avg"] = result["_weighted_sentiment"] / result[
            "weight_sum"
        ].replace(0, 1)
        result = result.drop(columns=["_weighted_sentiment"])

    return result


def _create_model_panel(
    base_panel: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    config: RegressionConfig,
) -> pd.DataFrame:
    """Create panel with sentiment from a specific model.

    Takes the base panel (with financial data, targets, etc.) and replaces
    sentiment columns with data from the specified model.
    """
    from merging.features import standardize

    # Columns to drop (will be replaced with model-specific ones)
    sentiment_cols = [
        "sentiment_avg",
        "sentiment",
        "valid_sentiment",
        "sentiment_std",
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_pos_std",
        "sentiment_neg_std",
        "sentiment_disp",
        "sentiment_disp_std",
        "sentiment_intensity",
        "sentiment_intensity_std",
        "sentiment_x_attention",
        "sentiment_pos_x_attention",
        "sentiment_neg_x_attention",
    ]

    # Also drop lagged predictor columns (will be re-created)
    lag_cols_to_drop = [c for c in base_panel.columns if "_lag" in c]
    drop_cols = sentiment_cols + lag_cols_to_drop

    # Keep columns not related to sentiment
    keep_cols = [c for c in base_panel.columns if c not in drop_cols]
    panel = base_panel[keep_cols].copy()

    # Rename sentiment_df columns for merge
    sentiment_df = sentiment_df.rename(columns={"trading_date": "date"})

    # Merge new sentiment data
    panel = panel.merge(
        sentiment_df[
            ["date", "ticker", "sentiment_avg", "mention_count"]
            + (["sentiment_disp"] if "sentiment_disp" in sentiment_df.columns else [])
        ],
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_new"),
    )

    # Handle column conflicts
    if "mention_count_new" in panel.columns:
        panel["mention_count"] = panel["mention_count_new"].fillna(
            panel.get("mention_count", 0)
        )
        panel = panel.drop(columns=["mention_count_new"])

    # Fill missing sentiment
    panel["mention_count"] = panel["mention_count"].fillna(0).astype(int)
    panel["sentiment_avg"] = panel["sentiment_avg"].fillna(0.0)
    if "sentiment_disp" in panel.columns:
        panel["sentiment_disp"] = panel["sentiment_disp"].fillna(0.0)

    # Compute sentiment variables
    min_posts = config.sentiment.min_posts_for_valid_sentiment
    std_scope = config.sentiment.standardization

    # Valid sentiment indicator
    panel["valid_sentiment"] = (panel["mention_count"] >= min_posts).astype(int)

    # Adjusted sentiment
    panel["sentiment"] = np.where(
        panel["valid_sentiment"] == 1,
        panel["sentiment_avg"],
        0.0,
    )

    # Aggregate sentiment (standardized)
    if std_scope == "within_stock":
        panel["sentiment_std"] = standardize(
            panel["sentiment"], scope="within_stock", group_col=panel["ticker"]
        )
    else:
        panel["sentiment_std"] = standardize(panel["sentiment"], scope="global")

    # Positive/negative split
    panel["sentiment_pos"] = np.maximum(panel["sentiment"], 0.0)
    panel["sentiment_neg"] = np.minimum(panel["sentiment"], 0.0)

    if std_scope == "within_stock":
        panel["sentiment_pos_std"] = standardize(
            panel["sentiment_pos"], scope="within_stock", group_col=panel["ticker"]
        )
        panel["sentiment_neg_std"] = standardize(
            panel["sentiment_neg"], scope="within_stock", group_col=panel["ticker"]
        )
    else:
        panel["sentiment_pos_std"] = standardize(panel["sentiment_pos"], scope="global")
        panel["sentiment_neg_std"] = standardize(panel["sentiment_neg"], scope="global")

    # Interaction terms
    if "attention_std" in panel.columns:
        panel["sentiment_x_attention"] = panel["sentiment_std"] * panel["attention_std"]
        panel["sentiment_pos_x_attention"] = (
            panel["sentiment_pos_std"] * panel["attention_std"]
        )
        panel["sentiment_neg_x_attention"] = (
            panel["sentiment_neg_std"] * panel["attention_std"]
        )

    # Re-compute lagged predictors for the new sentiment columns
    from merging.features import compute_lagged_predictors

    panel = compute_lagged_predictors(panel, config.horizons)

    return panel
