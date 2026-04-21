"""Feature engineering for regression analysis.

This module implements variable construction for the backward-looking
volatility regression framework:
- Attention variables (log-transformed, standardized)
- Sentiment variables (positive/negative split, validity indicator)
- Rolling volatility target (backward-looking RMS of daily variances)
- Lagged predictors (all regressors shifted by h days)
- Interaction terms

All variables use trading-day indexing with proper weekend/holiday handling.

Backward-looking notation:
    ln V̄^(h)_{i,t} = α_i + δ_t + γ₁ Ã_{i,t-h} + ... + u_{i,t}
    where V̄^(h)_{i,t} = sqrt( (1/h) Σ_{k=0}^{h-1} σ²_{i,t-k} )
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from config import RegressionConfig

# =============================================================================
# Standardization Utilities
# =============================================================================


def standardize(
    series: pd.Series,
    scope: str = "global",
    group_col: pd.Series | None = None,
) -> pd.Series:
    """Standardize a series using z-scores.

    Args:
        series: Values to standardize
        scope: "global" for pooled standardization, "within_stock" for per-group
        group_col: Grouping column (required if scope="within_stock")

    Returns:
        Standardized series (z-scores)
    """
    if scope == "global":
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std

    elif scope == "within_stock":
        if group_col is None:
            raise ValueError("group_col required for within_stock standardization")

        # Create DataFrame for groupby operation
        df = pd.DataFrame({"value": series, "group": group_col})

        def z_score(x):
            if len(x) <= 1 or x.std() == 0:
                return pd.Series(0.0, index=x.index)
            return (x - x.mean()) / x.std()

        result = df.groupby("group")["value"].transform(z_score)
        return result

    else:
        raise ValueError(f"Unknown standardization scope: {scope}")


# =============================================================================
# Attention Variables
# =============================================================================


def compute_attention_variables(
    df: pd.DataFrame,
    config: RegressionConfig,
) -> pd.DataFrame:
    """Compute attention variables from mention counts.

    Implements Section 4.5.1 of Regression.tex:
    - A^abs = log(1 + n_{i,t})  [absolute]
    - A^rel = n_{i,t} / sum_j(n_{j,t})  [relative share]
    - Ã = (A - μ_A) / σ_A  [standardized]

    Args:
        df: DataFrame with columns [date, ticker, mention_count]
        config: Regression configuration

    Returns:
        DataFrame with added attention columns:
        - attention: Raw attention measure
        - attention_std: Standardized attention (z-score)
    """
    df = df.copy()
    measure = config.attention.measure
    std_scope = config.attention.standardization

    # Ensure mention_count exists
    if "mention_count" not in df.columns:
        df["mention_count"] = 0

    # Fill NaN mention counts with 0
    df["mention_count"] = df["mention_count"].fillna(0).astype(int)

    # Compute raw attention measure
    if measure == "absolute":
        # A^abs = log(1 + n_{i,t})
        df["attention"] = np.log1p(df["mention_count"])

    elif measure == "relative":
        # A^rel = n_{i,t} / sum_j(n_{j,t})
        # Sum of mentions per date
        daily_total = df.groupby("date")["mention_count"].transform("sum")
        # Avoid division by zero
        daily_total = daily_total.replace(0, 1)
        df["attention"] = df["mention_count"] / daily_total

    else:
        raise ValueError(f"Unknown attention measure: {measure}")

    # Standardize
    if std_scope == "within_stock":
        df["attention_std"] = standardize(
            df["attention"],
            scope="within_stock",
            group_col=df["ticker"],
        )
    else:
        df["attention_std"] = standardize(df["attention"], scope="global")

    return df


# =============================================================================
# Sentiment Variables
# =============================================================================


def compute_sentiment_variables(
    df: pd.DataFrame,
    config: RegressionConfig,
) -> pd.DataFrame:
    """Compute sentiment variables from daily sentiment scores.

    Implements Section 4.5.2 of Regression.tex:
    - S_{i,t} = mean sentiment score
    - S* = 0 if mention_count < threshold (invalid)
    - S+ = max(S*, 0), S- = min(S*, 0)
    - ValidSent = 1 if mention_count >= threshold

    Args:
        df: DataFrame with columns [date, ticker, sentiment_avg, sentiment_disp, mention_count]
        config: Regression configuration

    Returns:
        DataFrame with added sentiment columns:
        - sentiment: Adjusted sentiment (0 for low-volume days)
        - sentiment_pos: Positive component max(S, 0)
        - sentiment_neg: Negative component min(S, 0)
        - sentiment_pos_std, sentiment_neg_std: Standardized versions
        - valid_sentiment: 1 if mention_count >= threshold
        - sentiment_disp: Dispersion (if available)
        - sentiment_intensity: |S| (if enabled)
    """
    df = df.copy()
    min_posts = config.sentiment.min_posts_for_valid_sentiment
    std_scope = config.sentiment.standardization

    # Ensure required columns exist
    if "sentiment_avg" not in df.columns:
        df["sentiment_avg"] = 0.0
    if "mention_count" not in df.columns:
        df["mention_count"] = 0

    # Valid sentiment indicator
    df["valid_sentiment"] = (df["mention_count"] >= min_posts).astype(int)

    # Adjusted sentiment: set to 0 for low-volume days
    df["sentiment"] = np.where(
        df["valid_sentiment"] == 1,
        df["sentiment_avg"],
        0.0,
    )

    # Always create BOTH parameterizations:
    # 1. Aggregate sentiment (for extension regression)
    # 2. Positive/negative split (for baseline regression)

    # Aggregate sentiment (needed for extension)
    if std_scope == "within_stock":
        df["sentiment_std"] = standardize(
            df["sentiment"],
            scope="within_stock",
            group_col=df["ticker"],
        )
    else:
        df["sentiment_std"] = standardize(df["sentiment"], scope="global")

    # Positive/negative split (needed for baseline)
    df["sentiment_pos"] = np.maximum(df["sentiment"], 0.0)
    df["sentiment_neg"] = np.minimum(df["sentiment"], 0.0)

    if std_scope == "within_stock":
        df["sentiment_pos_std"] = standardize(
            df["sentiment_pos"],
            scope="within_stock",
            group_col=df["ticker"],
        )
        df["sentiment_neg_std"] = standardize(
            df["sentiment_neg"],
            scope="within_stock",
            group_col=df["ticker"],
        )
    else:
        df["sentiment_pos_std"] = standardize(df["sentiment_pos"], scope="global")
        df["sentiment_neg_std"] = standardize(df["sentiment_neg"], scope="global")

    # Sentiment dispersion (standard deviation of post sentiments)
    if config.sentiment.include_dispersion and "sentiment_disp" in df.columns:
        # Fill NaN dispersion with 0 for low-volume days
        df["sentiment_disp"] = df["sentiment_disp"].fillna(0.0)

        if std_scope == "within_stock":
            df["sentiment_disp_std"] = standardize(
                df["sentiment_disp"],
                scope="within_stock",
                group_col=df["ticker"],
            )
        else:
            df["sentiment_disp_std"] = standardize(df["sentiment_disp"], scope="global")

    # Sentiment intensity (absolute value)
    if config.sentiment.include_intensity:
        df["sentiment_intensity"] = np.abs(df["sentiment"])

        if std_scope == "within_stock":
            df["sentiment_intensity_std"] = standardize(
                df["sentiment_intensity"],
                scope="within_stock",
                group_col=df["ticker"],
            )
        else:
            df["sentiment_intensity_std"] = standardize(
                df["sentiment_intensity"], scope="global"
            )

    return df


# =============================================================================
# Daily Variance Measures
# =============================================================================

# Registry of supported daily volatility measures.
# Maps the measure key (used in regression specs and output naming) to the
# corresponding daily-volatility column produced upstream by
# financial_data/returns.py.
#
#   ja_gk : baseline    — jump-adjusted Garman--Klass (1980)
#   ja_rs : robustness  — jump-adjusted Rogers--Satchell (1991)
#   ja_pk : robustness  — jump-adjusted Parkinson (1980)
#   gk    : robustness  — pure Garman--Klass
#   pk    : robustness  — pure Parkinson
#   sqret : robustness  — |r_t|, i.e. squared close-to-close log return
VOLATILITY_MEASURES = {
    "ja_gk": "volatility_ja_gk",
    "ja_rs": "volatility_ja_rs",
    "ja_pk": "volatility_ja_pk",
    "gk": "volatility_gk",
    "pk": "volatility_pk",
    "sqret": "volatility_sqret",
}

# Mapping from measure key to the corresponding log-variance column produced
# by compute_lagged_controls() (used as the lagged-volatility regressor).
LOG_VARIANCE_COLUMNS = {key: f"log_variance_{key}" for key in VOLATILITY_MEASURES}


# =============================================================================
# Forward Returns and Volatility (Target Variables)
# =============================================================================


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """Compute forward h-day log returns.

    Implements Section 4.4.1 of Regression.tex:
    r_{i,t→t+h} = ln(P_{i,t+h}) - ln(P_{i,t})

    Args:
        df: DataFrame with columns [date, ticker, adj_close] sorted by [ticker, date]
        horizons: List of forecast horizons (e.g., [1, 3, 5])

    Returns:
        DataFrame with added columns fwd_return_{h}d for each horizon
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Log price for cumulative return calculation
    df["log_price"] = np.log(df["adj_close"])

    for h in horizons:
        # Forward log price (h trading days ahead)
        df[f"log_price_fwd_{h}"] = df.groupby("ticker")["log_price"].shift(-h)

        # Forward return: ln(P_{t+h}) - ln(P_t)
        df[f"fwd_return_{h}d"] = df[f"log_price_fwd_{h}"] - df["log_price"]

        # Clean up temporary column
        df = df.drop(columns=[f"log_price_fwd_{h}"])

    # Clean up
    df = df.drop(columns=["log_price"])

    return df


def compute_rolling_volatility(
    df: pd.DataFrame,
    horizons: List[int],
    aggregation: str = "mean",
    measure: str = "ja_gk",
) -> pd.DataFrame:
    """Compute backward-looking h-day RMS volatility.

    Implements the backward-looking target variable:
        V̄^(h)_{i,t} = sqrt( (1/h) Σ_{k=0}^{h-1} σ²_{i,t-k} )  [mean/RMS]
        V̄^(h)_{i,t} = sqrt( Σ_{k=0}^{h-1} σ²_{i,t-k} )         [sum]

    The regression target is ln(V̄) to handle right-skewness.

    Args:
        df: DataFrame with daily volatility columns, sorted by [ticker, date]
        horizons: List of horizons h (e.g., [1, 3, 5])
        aggregation: "mean" for RMS (default), "sum" for cumulative
        measure: Volatility measure key (one of VOLATILITY_MEASURES)

    Returns:
        DataFrame with added columns:
        - rolling_volatility_{h}d[_{suffix}]: V̄^(h)
        - log_volatility_{h}d[_{suffix}]: ln(V̄^(h)) — regression target
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Determine source column and output suffix
    vol_col = VOLATILITY_MEASURES[measure]
    suffix = "" if measure == "ja_gk" else f"_{measure}"

    # Square volatility once to get daily variances σ²_{i,t}
    df["_var_tmp"] = df[vol_col] ** 2

    for h in horizons:
        # Collect variances for the past h days: σ²_{t}, σ²_{t-1}, ..., σ²_{t-h+1}
        var_cols = []
        for k in range(h):
            col_name = f"_var_shift_{k}"
            df[col_name] = df.groupby("ticker")["_var_tmp"].shift(k)
            var_cols.append(col_name)

        # RMS: sqrt( (1/h) Σ σ² ) or sqrt( Σ σ² )
        if aggregation == "mean":
            df[f"rolling_volatility_{h}d{suffix}"] = np.sqrt(df[var_cols].mean(axis=1))
        else:  # sum
            df[f"rolling_volatility_{h}d{suffix}"] = np.sqrt(df[var_cols].sum(axis=1))

        # Handle NaN (not enough historical data at start of series)
        df.loc[df[var_cols].isna().any(axis=1), f"rolling_volatility_{h}d{suffix}"] = (
            np.nan
        )

        # Replace zeros with NaN before log
        df.loc[
            df[f"rolling_volatility_{h}d{suffix}"] == 0,
            f"rolling_volatility_{h}d{suffix}",
        ] = np.nan

        # Log volatility (main regression target)
        df[f"log_volatility_{h}d{suffix}"] = np.log(
            df[f"rolling_volatility_{h}d{suffix}"]
        )

        # Clean up temporary columns
        df = df.drop(columns=var_cols)

    df = df.drop(columns=["_var_tmp"])
    return df


# =============================================================================
# Lagged Controls
# =============================================================================


def compute_lagged_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged control variables.

    In the backward-looking notation, the lagged volatility control at
    the predictor time t-h is ln(σ²_{i,t-h}) — the log daily variance
    on the day the predictors are observed.

    Creates one log-variance control column per available volatility
    measure (see VOLATILITY_MEASURES) so that downstream regressions can
    use the measure-consistent control. Also computes the lagged daily
    return as an auxiliary base predictor.

    These "base" columns are computed here; the actual shifting by h
    happens in compute_lagged_predictors().

    Args:
        df: DataFrame containing the daily volatility columns produced by
            financial_data/returns.py and a log_return column.

    Returns:
        DataFrame with one log_variance_{measure} column per available
        measure, plus a lag_return column.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # One log-variance control per volatility measure.
    for measure, vol_col in VOLATILITY_MEASURES.items():
        if vol_col not in df.columns:
            continue
        var = df[vol_col] ** 2
        var = var.replace(0, np.nan)
        df[f"log_variance_{measure}"] = np.log(var)

    # Lagged return (previous day's return) — used as a base predictor
    df["lag_return"] = df.groupby("ticker")["log_return"].shift(1)

    return df


# =============================================================================
# Lagged Predictors (Backward-Looking Shift)
# =============================================================================


def compute_lagged_predictors(
    df: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """Shift all predictor variables by h days to implement backward-looking notation.

    In the backward-looking model, the target V̄^(h)_{i,t} is explained by
    predictors observed at time t-h. This function creates horizon-specific
    lagged columns for each predictor used in the regression.

    For each horizon h, the following lagged columns are created:
        {predictor}_lag{h} = {predictor} shifted forward by h rows (within ticker)

    Args:
        df: DataFrame with predictor columns, sorted by [ticker, date]
        horizons: List of horizons h (e.g., [1, 3, 5])

    Returns:
        DataFrame with added lag{h} columns for each predictor and horizon
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Predictor columns to lag
    predictor_cols = [
        # Attention
        "attention_std",
        # Sentiment (baseline: pos/neg split)
        "sentiment_pos_std",
        "sentiment_neg_std",
        # Sentiment (extension: aggregate)
        "sentiment_std",
        # Validity indicator
        "valid_sentiment",
        # Interaction terms (baseline)
        "sentiment_pos_x_attention",
        "sentiment_neg_x_attention",
        # Interaction terms (extension)
        "sentiment_x_attention",
        # Return control
        "lag_return",
    ]

    # Volatility controls: include log_variance for every available measure
    for measure in VOLATILITY_MEASURES:
        col = f"log_variance_{measure}"
        if col in df.columns:
            predictor_cols.append(col)

    # Optional columns (may not exist depending on config)
    optional_cols = [
        "sentiment_disp_std",
        "sentiment_intensity_std",
    ]

    for col in optional_cols:
        if col in df.columns:
            predictor_cols.append(col)

    for h in horizons:
        for col in predictor_cols:
            if col in df.columns:
                df[f"{col}_lag{h}"] = df.groupby("ticker")[col].shift(h)

    return df


# =============================================================================
# Interaction Terms
# =============================================================================


def compute_interaction_terms(
    df: pd.DataFrame,
    config: RegressionConfig,
) -> pd.DataFrame:
    """Compute interaction terms between sentiment and attention.

    Implements Section 4.6 of Regression.tex:
    - S+ × A (positive sentiment × attention)
    - S- × A (negative sentiment × attention)

    Uses standardized variables for interpretable coefficients.

    Args:
        df: DataFrame with standardized attention and sentiment columns
        config: Regression configuration

    Returns:
        DataFrame with added interaction columns
    """
    df = df.copy()

    # Always create BOTH types of interactions:
    # 1. S×A for extension regression
    # 2. S⁺×A, S⁻×A for baseline regression

    # Aggregate sentiment interaction (for extension)
    if "sentiment_std" in df.columns and "attention_std" in df.columns:
        df["sentiment_x_attention"] = df["sentiment_std"] * df["attention_std"]

    # Positive/negative sentiment interactions (for baseline)
    if "sentiment_pos_std" in df.columns and "attention_std" in df.columns:
        df["sentiment_pos_x_attention"] = df["sentiment_pos_std"] * df["attention_std"]
    if "sentiment_neg_std" in df.columns and "attention_std" in df.columns:
        df["sentiment_neg_x_attention"] = df["sentiment_neg_std"] * df["attention_std"]

    return df
