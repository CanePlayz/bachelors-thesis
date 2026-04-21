"""Merging package for panel data construction."""

from .builder import build_panel, load_panel, panel_exists
from .features import (
    compute_attention_variables,
    compute_forward_returns,
    compute_interaction_terms,
    compute_lagged_controls,
    compute_lagged_predictors,
    compute_rolling_volatility,
    compute_sentiment_variables,
)
from .io import load_panel_data, save_panel
from .loaders import (
    filter_stocks_by_criteria,
    get_trading_calendar,
    load_benchmark_data,
    load_returns_data,
    load_sentiment_data,
    map_calendar_to_trading_day,
)

__all__ = [
    # Loaders
    "load_returns_data",
    "load_benchmark_data",
    "load_sentiment_data",
    "get_trading_calendar",
    "map_calendar_to_trading_day",
    "filter_stocks_by_criteria",
    # Features
    "compute_attention_variables",
    "compute_sentiment_variables",
    "compute_forward_returns",
    "compute_rolling_volatility",
    "compute_lagged_controls",
    "compute_lagged_predictors",
    "compute_interaction_terms",
    # Builder
    "build_panel",
    "load_panel",
    "panel_exists",
    # I/O
    "save_panel",
    "load_panel_data",
]
