"""Configuration for the regression analysis pipeline.

This module provides centralized configuration for:
- Input/output paths
- Sentiment model selection (primary and robustness)
- Forecast horizons
- Attention and sentiment variable definitions
- Standardization settings
- Fixed effects and clustering options
- Sample inclusion criteria

All configuration parameters can be overridden via command-line arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

# =============================================================================
# Directory Structure
# =============================================================================

# Resolve paths relative to the workspace root
SCRIPT_DIR = Path(__file__).parent.resolve()
REGRESSION_DIR = SCRIPT_DIR.parent
DATA_DIR = REGRESSION_DIR.parent

# Output directory for regression results
OUTPUT_DIR = REGRESSION_DIR / "out"

# =============================================================================
# Input Paths
# =============================================================================

# Financial data (from financial_data pipeline)
RETURNS_PARQUET = DATA_DIR / "financial_data" / "out" / "stock_returns.parquet"
BENCHMARK_CSV = DATA_DIR / "financial_data" / "out" / "market_benchmark.csv"

# Sentiment data (from reddit pipeline)
SENTIMENT_CSV = (
    DATA_DIR / "reddit" / "out" / "stage3_aggregated" / "daily_sentiment.csv"
)

# Trading calendar (US NYSE/Nasdaq trading days)
# If not present, will be derived from financial data
TRADING_CALENDAR_CSV = DATA_DIR / "financial_data" / "out" / "trading_calendar.csv"

# =============================================================================
# Sentiment Models
# =============================================================================

# Available sentiment models from the reddit pipeline
SENTIMENT_MODELS = {
    "roberta_topic": "cardiffnlp/twitter-roberta-base-topic-sentiment-latest",
    "bertweet": "finiteautomata/bertweet-base-sentiment-analysis",
    "fintwit": "StephanAkkerman/FinTwitBERT-sentiment",
}

# Primary model for main regressions
PRIMARY_SENTIMENT_MODEL = "roberta_topic"

# Models used for robustness checks (all models except primary)
ROBUSTNESS_SENTIMENT_MODELS = ["bertweet", "fintwit"]

# =============================================================================
# Forecast Horizons
# =============================================================================

# Horizons (in trading days) for forward returns/volatility
# h=1 means predicting tomorrow's return using today's information
FORECAST_HORIZONS: List[int] = [1, 3, 5]

# =============================================================================
# Attention Variable Configuration
# =============================================================================


@dataclass
class AttentionConfig:
    """Configuration for attention variable construction.

    Attention measures social media activity for a stock on a given day.
    Two main approaches:
    1. Absolute: log(1 + mention_count) - captures raw activity level
    2. Relative: share of total daily mentions - captures salience

    Standardization:
    - Global: z-score across all stocks and dates (preserves cross-sectional comparability)
    - Within-stock: z-score within each stock (captures deviations from stock's baseline)

    The four combinations of (measure, standardization) yield four distinct
    attention specifications tested in Regression.tex Section 4.5.1.
    """

    # Which attention measure to use
    # "absolute": log(1 + n_{i,t})
    # "relative": n_{i,t} / sum_j(n_{j,t})
    measure: Literal["absolute", "relative"] = "absolute"

    # Standardization scope
    # "global": z-score across all (ticker, date) pairs
    # "within_stock": z-score within each ticker separately
    standardization: Literal["global", "within_stock"] = "within_stock"


# All four attention specifications for robustness
ATTENTION_ROBUSTNESS_SPECS = [
    # (measure, standardization) — primary is excluded at runtime
    AttentionConfig(measure="absolute", standardization="global"),
    AttentionConfig(measure="relative", standardization="within_stock"),
    AttentionConfig(measure="relative", standardization="global"),
]


# Default attention configuration
DEFAULT_ATTENTION_CONFIG = AttentionConfig()


# =============================================================================
# Sentiment Variable Configuration
# =============================================================================


@dataclass
class SentimentConfig:
    """Configuration for sentiment variable construction.

    Sentiment is computed as the weighted mean of post-level sentiment scores.
    Key choices:
    1. Split vs. aggregate: Separate positive/negative or single variable
    2. Validity threshold: Minimum posts for reliable sentiment estimate
    3. Standardization: Global vs. within-stock z-scores
    """

    # Whether to split sentiment into positive (S+) and negative (S-) components
    # If True: S+ = max(S, 0), S- = min(S, 0)
    # If False: Use aggregate sentiment S directly
    split_pos_neg: bool = True

    # Minimum number of posts required for valid sentiment
    # Days with fewer posts get sentiment = 0 and ValidSent = 0
    min_posts_for_valid_sentiment: int = 5

    # Standardization scope
    # "global": z-score across all (ticker, date) pairs
    # "within_stock": z-score within each ticker separately
    standardization: Literal["global", "within_stock"] = "within_stock"

    # Whether to include sentiment dispersion (std of post sentiments)
    include_dispersion: bool = True

    # Whether to include sentiment intensity (|S|)
    include_intensity: bool = True


# Default sentiment configuration
DEFAULT_SENTIMENT_CONFIG = SentimentConfig()


# =============================================================================
# Fixed Effects and Clustering
# =============================================================================


@dataclass
class EstimationConfig:
    """Configuration for panel regression estimation.

    Fixed effects control for:
    - Entity (stock) FE: Time-invariant firm characteristics
    - Time (date) FE: Market-wide shocks on each day

    Standard error clustering:
    - Entity: Serial correlation within stocks
    - Time: Cross-sectional correlation on each day
    - Two-way: Both (more conservative)
    """

    # Fixed effects specification
    # "none": Pooled OLS (no fixed effects) — useful to quantify how much
    #         of R² is absorbed by the FE structure
    # "entity": Stock fixed effects only (alpha_i)
    # "time": Date fixed effects only (delta_t)
    # "twoway": Both entity and time fixed effects
    fixed_effects: Literal["none", "entity", "time", "twoway"] = "twoway"

    # Standard error clustering
    # "entity": Cluster by stock
    # "time": Cluster by date
    # "twoway": Two-way clustering by stock and date
    cluster_se: Literal["entity", "time", "twoway"] = "entity"

    # Whether to use HAC (heteroskedasticity and autocorrelation consistent) SEs
    # within clusters. Adds Newey-West type correction.
    use_hac: bool = False


# Default estimation configuration
DEFAULT_ESTIMATION_CONFIG = EstimationConfig()


# =============================================================================
# Sample Inclusion Criteria
# =============================================================================


@dataclass
class SampleConfig:
    """Configuration for sample selection and filtering.

    Stocks are included in the sample if they satisfy minimum data requirements.
    This ensures sufficient statistical power and data quality.
    """

    # Minimum trading days with available financial data (price, volume)
    min_trading_days_financial: int = 200

    # Minimum trading days with at least one social media mention
    min_trading_days_mentioned: int = 30

    # Minimum trading days with valid sentiment (>= min_posts threshold)
    min_trading_days_valid_sentiment: int = 20

    # Date range for analysis (inclusive)
    # Format: "YYYY-MM-DD"
    # If None, uses full available range
    start_date: str | None = "2018-01-01"
    end_date: str | None = "2024-12-31"


# Default sample configuration
DEFAULT_SAMPLE_CONFIG = SampleConfig()


# =============================================================================
# Volatility Target Configuration
# =============================================================================


@dataclass
class VolatilityConfig:
    """Configuration for backward-looking volatility target construction.

    The target variable V̄^(h)_{i,t} is defined as the root-mean-square (RMS)
    of daily variances over the past h days:

        V̄^(h)_{i,t} = sqrt( (1/h) Σ_{k=0}^{h-1} σ²_{i,t-k} )

    The regression then predicts ln V̄^(h)_{i,t} using predictors from day t-h.

    Six daily volatility measures are supported.  The baseline is the
    jump-adjusted Garman--Klass (JA-GK) estimator, which decomposes the daily
    variance into its overnight (close-to-open jump) and intraday components
    in the spirit of Yang and Zhang (2000) and uses the Garman--Klass (1980)
    range estimator for the intraday term — the most efficient range-based
    intraday estimator under standard assumptions (Molnár, 2012).

    Measures:
    - ja_gk: jump-adjusted Garman--Klass (baseline)
    - ja_rs: jump-adjusted Rogers--Satchell (1991)
    - ja_pk: jump-adjusted Parkinson (1980)
    - gk:    pure Garman--Klass (no overnight component)
    - pk:    pure Parkinson (no overnight component)
    - sqret: |r_t|, squared close-to-close log return as variance proxy
    """

    # Primary volatility measure for baseline regressions
    measure: Literal["ja_gk", "ja_rs", "ja_pk", "gk", "pk", "sqret"] = "ja_gk"

    # Alternative measures for robustness checks
    robustness_measures: List[str] = field(
        default_factory=lambda: ["ja_rs", "ja_pk", "gk", "pk", "sqret"]
    )

    # How to aggregate variance across the horizon window
    # "mean": V̄ = sqrt( (1/h) Σ σ² )  — RMS, comparable across horizons (DEFAULT)
    # "sum":  V̄ = sqrt( Σ σ² )         — cumulative volatility exposure
    aggregation: Literal["sum", "mean"] = "mean"


# Default volatility configuration
DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()


# =============================================================================
# Output Configuration
# =============================================================================


@dataclass
class OutputConfig:
    """Configuration for output file generation.

    Results are organized in subdirectories by regression type:
    - out/panel/: Panel data files
    - out/regression/raw/: Raw regression results (JSON, CSV)
    - out/regression/tables/: LaTeX tables
    """

    # Output directory (relative to REGRESSION_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    # Whether to save panel data
    save_panel: bool = True

    # Output formats for regression results
    save_json: bool = True  # Full results with all statistics
    save_csv: bool = True  # Summary table format
    save_latex: bool = True  # Publication-ready tables

    # Whether to generate summary statistics
    save_summary_stats: bool = True


# Default output configuration
DEFAULT_OUTPUT_CONFIG = OutputConfig()


# =============================================================================
# Aggregate Configuration
# =============================================================================


@dataclass
class RegressionConfig:
    """Complete configuration for regression analysis.

    Combines all sub-configurations into a single object.
    """

    # Primary sentiment model
    primary_sentiment_model: str = PRIMARY_SENTIMENT_MODEL

    # Robustness models
    robustness_sentiment_models: List[str] = field(
        default_factory=lambda: ROBUSTNESS_SENTIMENT_MODELS.copy()
    )

    # Forecast horizons
    horizons: List[int] = field(default_factory=lambda: FORECAST_HORIZONS.copy())

    # Sub-configurations
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.primary_sentiment_model not in SENTIMENT_MODELS:
            raise ValueError(
                f"Unknown primary sentiment model: {self.primary_sentiment_model}. "
                f"Available: {list(SENTIMENT_MODELS.keys())}"
            )

        for model in self.robustness_sentiment_models:
            if model not in SENTIMENT_MODELS:
                raise ValueError(
                    f"Unknown robustness sentiment model: {model}. "
                    f"Available: {list(SENTIMENT_MODELS.keys())}"
                )

        if not self.horizons:
            raise ValueError("At least one forecast horizon must be specified")

        for h in self.horizons:
            if h < 1:
                raise ValueError(f"Forecast horizon must be >= 1, got {h}")


# Default configuration instance
DEFAULT_CONFIG = RegressionConfig()
