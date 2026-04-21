"""Regression package for panel estimation."""

from .estimation import run_panel_regression, run_regression_spec
from .runner import (
    run_all_regressions,
    run_baseline_regressions,
    run_extension_regressions,
)
from .specs import (
    RegressionResult,
    RegressionSpec,
    create_baseline_volatility_spec,
    create_extension_volatility_spec,
)

__all__ = [
    # Specifications
    "RegressionSpec",
    "RegressionResult",
    "create_baseline_volatility_spec",
    "create_extension_volatility_spec",
    # Estimation
    "run_panel_regression",
    "run_regression_spec",
    # Runner
    "run_all_regressions",
    "run_baseline_regressions",
    "run_extension_regressions",
]
