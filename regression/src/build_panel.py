"""Build panel data for regression analysis.

Usage:
    python build_panel.py            # Build with default config
    python build_panel.py --force    # Force rebuild

The panel is always built with the primary sentiment model and the primary
attention specification. Robustness variants (alternative sentiment models,
attention specs, volatility measures) are swapped in at runtime by
``run_regression.py`` and therefore do not need their own panel build.

Output:
    out/panel/panel_data.parquet
    out/panel/panel_metadata.json
"""

from __future__ import annotations

import argparse
import sys
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

from config import OUTPUT_DIR, RegressionConfig
from merging.builder import build_panel


def main() -> int:
    """Main entry point for building panel data."""
    parser = argparse.ArgumentParser(
        description="Build panel data for regression analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if panel exists",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "panel",
        help=f"Output directory (default: {OUTPUT_DIR / 'panel'})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PANEL DATA BUILDER")
    print("=" * 60)

    config = RegressionConfig()

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    print("\nConfiguration:")
    print(f"  Sentiment model: {config.primary_sentiment_model}")
    print(f"  Horizons: {config.horizons}")
    print(
        f"  Attention: {config.attention.measure} ({config.attention.standardization})"
    )
    print(f"  Volatility aggregation: {config.volatility.aggregation}")
    print(f"  Date range: {config.sample.start_date} to {config.sample.end_date}")
    print(f"  Output: {args.output_dir}")

    try:
        _, panel_path = build_panel(
            config=config,
            force=args.force,
            output_dir=args.output_dir,
        )
        print(f"\nPanel built successfully: {panel_path}")
        return 0

    except Exception as e:
        print(f"\nERROR building panel: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
