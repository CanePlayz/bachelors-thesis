"""Configuration and path constants for financial data fetching.

This module provides centralized configuration for:
- File paths (input/output locations)
- Trading calendar constants

The fetched ticker universe and the date range are both auto-discovered from
the Reddit pipeline outputs (mention stats); there are no manual defaults.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Directory Structure
# =============================================================================

# Directory containing this file
SCRIPT_DIR = Path(__file__).parent.resolve()
# Module root (financial_data/), one level above src/
MODULE_DIR = SCRIPT_DIR.parent
# Repository root
DATA_DIR = MODULE_DIR.parent

# Output directory
OUTPUT_DIR = MODULE_DIR / "out"

# =============================================================================
# Input Paths (from Reddit sentiment pipeline)
# =============================================================================

# Path to the curated ticker universe (used to filter out ETFs).
TICKERS_CSV = DATA_DIR / "reddit" / "tickers" / "clean_tickers.csv"

# Path to global mention stats (one row per mentioned ticker).
# Used to determine which tickers to fetch financial data for.
MENTION_STATS_CSV = (
    DATA_DIR / "reddit" / "out" / "stats" / "global" / "aggregated" / "ticker_stats.csv"
)

# Path to global daily mention totals (one row per date).
# Used to auto-discover the date range of the Reddit pipeline output.
MENTION_DAILY_TOTALS_CSV = (
    DATA_DIR / "reddit" / "out" / "stats" / "global" / "aggregated" / "daily_totals.csv"
)

# =============================================================================
# Market Configuration
# =============================================================================

# Market benchmark ticker (S&P 500)
MARKET_BENCHMARK = "^GSPC"

# =============================================================================
# Trading Calendar Constants
# =============================================================================

TRADING_DAYS_PER_YEAR = 252  # Standard assumption for annualization
