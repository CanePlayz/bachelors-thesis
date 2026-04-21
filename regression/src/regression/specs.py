"""Regression specification and result types.

This module defines:
- RegressionSpec: Specification for a single regression
- RegressionResult: Container for regression results
- Factory functions for creating standard specifications

The specifications mirror Sections 5.1–5.2 of the thesis:
- Baseline volatility regression (Equation 5.2)
- Extension with magnitude and dispersion (Equation 5.4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class RegressionSpec:
    """Specification for a panel regression.

    Attributes:
        name: Descriptive name for the regression
        target: Dependent variable column name
        regressors: List of regressor column names
        entity_effects: Include stock fixed effects (α_i)
        time_effects: Include date fixed effects (δ_t)
        cluster: Standard error clustering method
        horizon: Forecast horizon (for labeling)
        description: Human-readable description
    """

    name: str
    target: str
    regressors: List[str]
    entity_effects: bool = True
    time_effects: bool = True
    cluster: Literal["entity", "time", "twoway"] = "entity"
    horizon: int = 1
    description: str = ""

    def get_formula_str(self) -> str:
        """Get a string representation of the regression formula."""
        fe_str = ""
        if self.entity_effects and self.time_effects:
            fe_str = " + α_i + δ_t"
        elif self.entity_effects:
            fe_str = " + α_i"
        elif self.time_effects:
            fe_str = " + δ_t"

        regressors_str = " + ".join(self.regressors)
        return f"{self.target} = {regressors_str}{fe_str} + ε"


@dataclass
class RegressionResult:
    """Container for regression results.

    Attributes:
        spec: The regression specification
        coefficients: Coefficient estimates
        std_errors: Standard errors (clustered)
        t_stats: t-statistics
        p_values: p-values
        conf_int_lower: 95% CI lower bounds
        conf_int_upper: 95% CI upper bounds
        r_squared: Overall R²
        r_squared_within: Within R² (after absorbing FE)
        n_obs: Number of observations
        n_entities: Number of stocks
        n_time: Number of time periods
        f_stat: F-statistic for joint significance
        f_pvalue: p-value for F-test
        model_info: Additional model information
    """

    spec: RegressionSpec
    coefficients: Dict[str, float] = field(default_factory=dict)
    std_errors: Dict[str, float] = field(default_factory=dict)
    t_stats: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    conf_int_lower: Dict[str, float] = field(default_factory=dict)
    conf_int_upper: Dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    r_squared_within: float = 0.0
    n_obs: int = 0
    n_entities: int = 0
    n_time: int = 0
    f_stat: float = 0.0
    f_pvalue: float = 1.0
    model_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "spec": {
                "name": self.spec.name,
                "target": self.spec.target,
                "regressors": self.spec.regressors,
                "entity_effects": self.spec.entity_effects,
                "time_effects": self.spec.time_effects,
                "cluster": self.spec.cluster,
                "horizon": self.spec.horizon,
                "description": self.spec.description,
                "formula": self.spec.get_formula_str(),
            },
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "t_stats": self.t_stats,
            "p_values": self.p_values,
            "conf_int_lower": self.conf_int_lower,
            "conf_int_upper": self.conf_int_upper,
            "r_squared": self.r_squared,
            "r_squared_within": self.r_squared_within,
            "n_obs": self.n_obs,
            "n_entities": self.n_entities,
            "n_time": self.n_time,
            "f_stat": self.f_stat,
            "f_pvalue": self.f_pvalue,
            "model_info": self.model_info,
        }

    def to_summary_row(self) -> Dict[str, Any]:
        """Convert to a single summary row for CSV output."""
        row = {
            "name": self.spec.name,
            "target": self.spec.target,
            "horizon": self.spec.horizon,
            "n_obs": self.n_obs,
            "n_entities": self.n_entities,
            "r_squared": self.r_squared,
            "r_squared_within": self.r_squared_within,
        }

        # Add coefficients with significance stars
        for var in self.spec.regressors:
            if var in self.coefficients:
                coef = self.coefficients[var]
                pval = self.p_values.get(var, 1.0)
                stars = _significance_stars(pval)
                row[f"coef_{var}"] = coef
                row[f"se_{var}"] = self.std_errors.get(var, None)
                row[f"pval_{var}"] = pval
                row[f"sig_{var}"] = stars

        return row


def _significance_stars(p: float) -> str:
    """Return significance stars based on p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    else:
        return ""


# =============================================================================
# Factory Functions for Standard Specifications
# =============================================================================


def create_baseline_volatility_spec(
    horizon: int,
    split_pos_neg: bool = True,
    entity_effects: bool = True,
    time_effects: bool = True,
    cluster: Literal["entity", "time", "twoway"] = "entity",
    vol_measure: str = "ja_gk",
) -> RegressionSpec:
    """Create baseline volatility regression specification.

    Backward-looking notation:
    ln V̄^(h)_{i,t} = α_i + δ_t + γ₁Ã_{t-h} + γ₂S̃⁺_{t-h} + γ₃S̃⁻_{t-h}
                     + γ₄(S̃⁺×Ã)_{t-h} + γ₅(S̃⁻×Ã)_{t-h}
                     + γ₆ValidSent_{t-h} + γ₇ln σ²_{t-h} + u

    Args:
        horizon: Horizon h (window size for rolling volatility)
        split_pos_neg: Use positive/negative sentiment split
        entity_effects: Include stock fixed effects
        time_effects: Include time fixed effects
        cluster: SE clustering method
        vol_measure: Volatility measure key. One of
            "ja_gk" (baseline: jump-adjusted Garman--Klass),
            "ja_rs" (jump-adjusted Rogers--Satchell),
            "ja_pk" (jump-adjusted Parkinson),
            "gk"    (pure Garman--Klass),
            "pk"    (pure Parkinson),
            "sqret" (squared close-to-close log return).

    Returns:
        RegressionSpec for baseline volatility regression
    """
    # Allowed measures and their corresponding lagged-variance regressor.
    # Each entry maps the measure key to the lag column name template.
    lag_var_template = {
        "ja_gk": "log_variance_ja_gk_lag{h}",
        "ja_rs": "log_variance_ja_rs_lag{h}",
        "ja_pk": "log_variance_ja_pk_lag{h}",
        "gk": "log_variance_gk_lag{h}",
        "pk": "log_variance_pk_lag{h}",
        "sqret": "log_variance_sqret_lag{h}",
    }
    if vol_measure not in lag_var_template:
        raise ValueError(f"Unknown volatility measure: {vol_measure}")

    # Determine target and control column names based on measure
    vol_suffix = "" if vol_measure == "ja_gk" else f"_{vol_measure}"
    target = f"log_volatility_{horizon}d{vol_suffix}"
    h = horizon

    lag_var_col = lag_var_template[vol_measure].format(h=h)

    if split_pos_neg:
        regressors = [
            f"attention_std_lag{h}",  # γ₁: Attention at t-h
            f"sentiment_pos_std_lag{h}",  # γ₂: Positive sentiment at t-h
            f"sentiment_neg_std_lag{h}",  # γ₃: Negative sentiment at t-h
            f"sentiment_pos_x_attention_lag{h}",  # γ₄: S⁺ × A at t-h
            f"sentiment_neg_x_attention_lag{h}",  # γ₅: S⁻ × A at t-h
            f"valid_sentiment_lag{h}",  # γ₆: Valid sentiment at t-h
            lag_var_col,  # γ₇: ln σ²_{i,t-h}
        ]
    else:
        regressors = [
            f"attention_std_lag{h}",  # γ₁: Attention at t-h
            f"sentiment_std_lag{h}",  # γ₂: Aggregate sentiment at t-h
            f"sentiment_x_attention_lag{h}",  # γ₃: S × A at t-h
            f"valid_sentiment_lag{h}",  # γ₄: Valid sentiment at t-h
            lag_var_col,  # γ₅: ln σ²_{i,t-h}
        ]

    # Name includes measure for robustness identification
    name_suffix = f"_{vol_measure}" if vol_measure != "ja_gk" else ""

    return RegressionSpec(
        name=f"volatility_h{horizon}{name_suffix}",
        target=target,
        regressors=regressors,
        entity_effects=entity_effects,
        time_effects=time_effects,
        cluster=cluster,
        horizon=horizon,
        description=f"Baseline volatility ({vol_measure}), h={horizon} trading days",
    )


def create_extension_volatility_spec(
    horizon: int,
    entity_effects: bool = True,
    time_effects: bool = True,
    cluster: Literal["entity", "time", "twoway"] = "entity",
) -> RegressionSpec:
    """Create extended volatility regression with magnitude and dispersion.

    Backward-looking notation with additional regressors:
    ln V̄^(h)_{i,t} = ... + γ₈ D̃_{t-h} + γ₉ |S̃_{t-h}| + u

    This uses a DIFFERENT parameterization than the baseline:
    - Baseline: S⁺, S⁻ → captures asymmetric effects of positive/negative sentiment
    - Extension: S (aggregate), |S| (magnitude), D (dispersion)

    Args:
        horizon: Horizon h
        entity_effects: Include stock fixed effects
        time_effects: Include time fixed effects
        cluster: SE clustering method

    Returns:
        RegressionSpec for extended volatility regression
    """
    target = f"log_volatility_{horizon}d"
    h = horizon

    regressors = [
        f"attention_std_lag{h}",  # γ₁: Attention at t-h
        f"sentiment_std_lag{h}",  # γ₂: Aggregate sentiment at t-h
        f"sentiment_x_attention_lag{h}",  # γ₃: S × A at t-h
        f"valid_sentiment_lag{h}",  # γ₄: Valid sentiment at t-h
        f"log_variance_ja_gk_lag{h}",  # γ₅: ln σ²_{i,t-h}
        f"sentiment_intensity_std_lag{h}",  # γ₆: |S̃| at t-h
        f"sentiment_disp_std_lag{h}",  # γ₇: D̃ at t-h
    ]

    return RegressionSpec(
        name=f"volatility_ext_h{horizon}",
        target=target,
        regressors=regressors,
        entity_effects=entity_effects,
        time_effects=time_effects,
        cluster=cluster,
        horizon=horizon,
        description=f"Extended volatility with magnitude/dispersion, h={horizon}",
    )
