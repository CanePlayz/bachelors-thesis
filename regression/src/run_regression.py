"""Run regression analysis on panel data.

Usage:
    python run_regression.py                   # Run all regressions
    python run_regression.py --skip-robustness # Skip robustness checks
    python run_regression.py --dry-run         # Show what would run

Output:
    out/regression/raw/{type}/             - Individual result files (JSON)
    out/regression/tables/                 - Summary tables (CSV, LaTeX)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 stdout/stderr on Windows so unicode characters in log output
# (arrows, R², etc.) don't crash under the default cp1252 codec.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_CONFIG, OUTPUT_DIR, SENTIMENT_MODELS, RegressionConfig
from merging.builder import build_panel, load_panel, panel_exists
from output.writer import save_all_results
from regression.runner import run_all_regressions


def main() -> int:
    """Main entry point for running regressions."""
    parser = argparse.ArgumentParser(
        description="Run regression analysis on panel data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_regression.py                   # Run all regressions
    python run_regression.py --skip-robustness # Skip robustness checks
    python run_regression.py --dry-run         # Show what would run
    python run_regression.py --force-panel     # Force panel rebuild first

Output:
    out/regression/raw/baseline_volatility/  - Volatility regression results
    out/regression/raw/extension_volatility/ - Extended volatility results
    out/regression/raw/robustness_*/         - Robustness check results
    out/regression/tables/                   - Summary tables
        """,
    )

    parser.add_argument(
        "--force-panel",
        action="store_true",
        help="Force rebuild panel before running regressions",
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=OUTPUT_DIR / "panel",
        help=f"Panel data directory (default: {OUTPUT_DIR / 'panel'})",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness regressions (alternative sentiment models, "
        "volatility measures, attention specifications)",
    )
    parser.add_argument(
        "--fixed-effects",
        choices=["none", "entity", "time", "twoway"],
        default=DEFAULT_CONFIG.estimation.fixed_effects,
        help=f"Fixed effects type (default: {DEFAULT_CONFIG.estimation.fixed_effects}). "
        f"Use 'none' for pooled OLS to quantify R² absorbed by FE.",
    )
    parser.add_argument(
        "--cluster",
        choices=["entity"],
        default=DEFAULT_CONFIG.estimation.cluster_se,
        help=f"SE clustering (default: {DEFAULT_CONFIG.estimation.cluster_se})",
    )
    parser.add_argument(
        "--model",
        choices=list(SENTIMENT_MODELS.keys()),
        default=DEFAULT_CONFIG.primary_sentiment_model,
        help=f"Primary sentiment model (default: {DEFAULT_CONFIG.primary_sentiment_model})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "regression",
        help=f"Output directory (default: {OUTPUT_DIR / 'regression'})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("REGRESSION ANALYSIS RUNNER")
    print("=" * 60)

    config = RegressionConfig(
        primary_sentiment_model=args.model,
    )
    config.estimation.fixed_effects = args.fixed_effects
    config.estimation.cluster_se = args.cluster

    print("\nConfiguration:")
    print(f"  Sentiment model: {config.primary_sentiment_model}")
    print(f"  Horizons: {config.horizons}")
    print(f"  Fixed effects: {config.estimation.fixed_effects}")
    print(f"  SE clustering: {config.estimation.cluster_se}")
    print(f"  Skip robustness: {args.skip_robustness}")

    # Tally regressions
    H = len(config.horizons)
    n_baseline = H
    n_extension = H
    n_robustness_sent = (
        0 if args.skip_robustness else H * len(config.robustness_sentiment_models)
    )
    n_robustness_vol = (
        0 if args.skip_robustness else H * len(config.volatility.robustness_measures)
    )
    n_robustness_att = 0 if args.skip_robustness else H * 3
    n_total = (
        n_baseline
        + n_extension
        + n_robustness_sent
        + n_robustness_vol
        + n_robustness_att
    )

    print("\nRegressions to run:")
    print(f"  Baseline volatility: {n_baseline}")
    print(f"  Extension volatility: {n_extension}")
    print(f"  Robustness (sentiment): {n_robustness_sent}")
    print(f"  Robustness (volatility): {n_robustness_vol}")
    print(f"  Robustness (attention): {n_robustness_att}")
    print(f"  Total: {n_total}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the above regressions")
        return 0

    if not panel_exists(args.panel_dir) or args.force_panel:
        print("\nBuilding panel data...")
        panel_df, _ = build_panel(
            config, force=args.force_panel, output_dir=args.panel_dir
        )
    else:
        print(f"\nLoading panel data from: {args.panel_dir}")
        panel_df = load_panel(args.panel_dir)
        print(f"  → {len(panel_df):,} rows, {panel_df['ticker'].nunique():,} tickers")

    start_time = time.time()

    results = run_all_regressions(
        panel=panel_df,
        config=config,
        output_dir=args.output_dir,
        skip_robustness=args.skip_robustness,
    )

    save_all_results(
        results=results,
        output_dir=args.output_dir,
        model_name=config.primary_sentiment_model,
    )

    elapsed = time.time() - start_time
    total_run = sum(len(v) for v in results.values())

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Regressions run: {total_run}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
