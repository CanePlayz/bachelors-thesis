"""Configuration for the GARCH-X estimation pipeline.

Joint panel-GARCH(1,1)-X estimation with Mean Group aggregation.

Model for stock i at time t:
    Mean:     r_{i,t} = eps_{i,t}               (zero-mean)
    Variance: sigma2_{i,t} = omega_i + alpha_i * eps2_{i,t-1}
                            + beta_i * sigma2_{i,t-1} + delta_i' * x_{i,t-1}
                            + gamma_i * VIX_{t-1}
                            + phi_i * gap_days_{t-1}
                            + psi_i * valid_sentiment_{i,t-1}

The valid_sentiment indicator (1 if the stock had >= min_posts mentions on day
t-1, 0 otherwise) mirrors the ValidSent control in the panel OLS specification.
It distinguishes "sentiment = 0 because posts were neutral" from "sentiment = 0
because no posts existed", which would otherwise be conflated by the NaN -> 0
fill for missing exogenous variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
GARCH_DIR = SCRIPT_DIR.parent
DATA_DIR = GARCH_DIR.parent

OUTPUT_DIR = GARCH_DIR / "out"
VIX_CACHE = GARCH_DIR / "out" / "vix_daily.parquet"

PANEL_PARQUET = DATA_DIR / "regression" / "out" / "panel" / "panel_data.parquet"


# =============================================================================
# Model Specification
# =============================================================================


@dataclass
class GARCHSpec:
    """Specification for panel GARCH(1,1)-X estimation."""

    exog_cols: List[str] = field(
        default_factory=lambda: [
            "attention_std",
            "sentiment_pos_std",
            "sentiment_neg_std",
            "vix_std",
            "gap_days",
            "valid_sentiment",
        ]
    )
    dist: Literal["normal"] = "normal"
    return_col: str = "log_return"
    verbose: bool = True

    # Parameter-pooling scopes used by the panel estimator and the LR
    # test runner (run_lr_tests.py). Defaults reflect the LR-test
    # selected configuration: alpha is globally pooled (LR did not
    # reject H0 at any conventional level), beta and delta are
    # stock-specific (LR rejects pooling). See run_lr_tests.py and the
    # methodology section of the paper for the test details.
    omega_scope: Literal["global", "stock"] = "stock"
    alpha_scope: Literal["global", "stock"] = "global"
    beta_scope: Literal["global", "stock"] = "stock"
    delta_scope: Literal["global", "stock"] = "stock"
    # Optional per-column override of delta_scope. When set, lists the names
    # of exogenous columns whose delta is *pooled* (a single shared value
    # across stocks); all other columns in exog_cols are stock-specific.
    # When None, delta_scope applies uniformly to all columns (legacy).
    delta_pooled_cols: Optional[List[str]] = None
    mu_scope: Literal["global", "stock", "zero"] = "zero"
    p: int = 1
    q: int = 1
    maxiter: int = 500
    optimizer: str = "L-BFGS-B"
    tol: float = 1e-8
    var_init: str = "unconditional"

    def is_delta_col_pooled(self, col_name: str) -> bool:
        """Return True if the delta on this exog column is globally pooled."""
        if self.delta_pooled_cols is not None:
            return col_name in self.delta_pooled_cols
        return self.delta_scope == "global"

    def describe(self) -> str:
        exog_str = ", ".join(self.exog_cols) if self.exog_cols else "(none)"
        if self.delta_pooled_cols is None:
            delta_scope_desc = self.delta_scope
        else:
            pooled = (
                ", ".join(self.delta_pooled_cols)
                if self.delta_pooled_cols
                else "(none)"
            )
            delta_scope_desc = f"mixed (pooled: {pooled})"
        return (
            f"Panel GARCH(1,1)-X (joint QML)\n"
            f"  Distribution: {self.dist}\n"
            f"  Exogenous:    {exog_str}\n"
            f"  Scopes:       alpha={self.alpha_scope}, beta={self.beta_scope}, delta={delta_scope_desc}"
        )


def default_spec() -> GARCHSpec:
    return GARCHSpec()


# =============================================================================
# Robustness Definitions
# =============================================================================

# Alternative sentiment models (robustness)
ROBUSTNESS_SENTIMENT_MODELS = ["bertweet", "fintwit"]

# Column suffixes for alternative sentiment models.
# The panel is expected to contain columns like
# ``sentiment_pos_std_bertweet``, ``attention_std`` (shared), etc.
ROBUSTNESS_SENTIMENT_COLUMN_MAP = {
    model: {
        "exog_cols": [
            "attention_std",  # attention is model-agnostic
            f"sentiment_pos_std_{model}",
            f"sentiment_neg_std_{model}",
            "vix_std",  # always included as CSD control
            "gap_days",  # backward-aggregation calendar control
            "valid_sentiment",  # missing-vs-neutral sentiment indicator
        ],
    }
    for model in ROBUSTNESS_SENTIMENT_MODELS
}

# Alternative attention specifications (measure × standardization)
# Primary is abs_within (already used in default spec).
ROBUSTNESS_ATTENTION_SPECS = [
    {"label": "abs_global", "col": "attention_std_abs_global"},
    {"label": "rel_within", "col": "attention_std_rel_within_stock"},
    {"label": "rel_global", "col": "attention_std_rel_global"},
]
