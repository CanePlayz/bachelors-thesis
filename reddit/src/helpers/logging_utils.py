"""Logging and formatting utilities for v2 pipeline (helpers).

This module provides utility functions for:
- Converting Unix timestamps to date strings with caching
- Formatting durations for display
- Printing fixed-width progress lines during pipeline execution

Key functions:
- timestamp_to_date(): Convert Unix timestamp to YYYY-MM-DD
- format_duration(): Format seconds as HH:MM:SS or MM:SS
- format_progress_line(): Generate a progress table row
- print_progress_header(): Print the progress table header
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

# =============================================================================
# Timestamp caching for fast date conversion
# =============================================================================

# Cache mapping day number (ts // 86400) -> date string
# This avoids redundant datetime conversions for same-day timestamps
_DATE_CACHE: Dict[int, str] = {}


# =============================================================================
# Indentation / banners (console output structure)
# =============================================================================


DEFAULT_INDENT = "  "


def indent(level: int = 1, unit: str = DEFAULT_INDENT) -> str:
    """Return an indentation prefix.

    Args:
        level: Indentation level (0 = no indent)
        unit: Indentation unit string (default: two spaces)

    Returns:
        Prefix string for console output.
    """
    if level <= 0:
        return ""
    return unit * level


def print_banner(title: str, prefix: str = "", width: int = 70) -> None:
    """Print a consistent section banner.

    Args:
        title: Banner title line.
        prefix: Optional left padding (used for nested sections).
        width: Banner line width (excluding prefix).
    """
    line = "=" * width
    print(f"\n{prefix}{line}")
    print(f"{prefix}{title}")
    print(f"{prefix}{line}")


def timestamp_to_date(ts: int) -> str:
    """Convert Unix timestamp to 'YYYY-MM-DD' string (UTC) with caching.

    Uses a day-based cache to avoid redundant datetime conversions.
    Since many Reddit posts occur on the same day, this provides
    significant speedup during batch processing.

    Args:
        ts: Unix timestamp (seconds since epoch)

    Returns:
        Date string in 'YYYY-MM-DD' format (UTC timezone)
    """
    # Handle invalid/zero timestamps
    if ts <= 0:
        return "1970-01-01"

    # Calculate day number for cache lookup
    # 86400 seconds = 1 day
    day = ts // 86400

    # Check cache first
    cached = _DATE_CACHE.get(day)
    if cached:
        return cached

    # Convert to datetime and format as date string
    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    # Store in cache for future lookups
    _DATE_CACHE[day] = date_str

    return date_str


# =============================================================================
# Duration formatting
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS for display.

    Provides human-readable duration strings for ETA and elapsed time.
    Uses zero-padded format for consistent column widths.

    Args:
        seconds: Duration in seconds (can be fractional)

    Returns:
        Formatted string like "05:30" or "01:23:45"
    """
    # Handle invalid negative values
    if seconds < 0:
        return "--:--"

    # Split into hours, minutes, seconds
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    # Include hours only if non-zero
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    # Default to MM:SS format
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# Progress line formatting
# =============================================================================


def format_progress_line(
    dataset: str,
    phase: str,
    processed: int,
    matches: int,
    rate: float,
    progress_pct: float,
    eta_seconds: float,
    prefix: str = "",
) -> str:
    """Format a fixed-width progress line with pipe separators.

    Creates a table row for the progress display with consistent
    column widths. Used to show real-time processing status.

    Args:
        dataset: Name of the dataset being processed
        phase: Current phase (e.g., "submissions", "comments")
        processed: Number of items processed so far
        matches: Number of items with ticker matches
        rate: Processing rate in items per second
        progress_pct: Completion percentage (0-100)
        eta_seconds: Estimated time remaining in seconds

    Returns:
        Formatted table row string like "| dataset | phase | ... |"
    """
    # Format each column with fixed widths
    # Truncate strings that exceed column width
    dataset_col = f"{dataset:<14}"[:14]  # Left-align, 14 chars
    phase_col = f"{phase:<11}"[:11]  # Left-align, 11 chars

    # Right-align numeric columns with thousands separators
    processed_col = f"{processed:>12,}"
    matches_col = f"{matches:>10,}"

    # Rate with /s suffix
    rate_col = f"{rate:>8,.0f}/s"

    # Percentage with 1 decimal place
    progress_col = f"{progress_pct:>5.1f}%"

    # ETA as formatted duration
    eta_col = f"{format_duration(eta_seconds):>8}"

    # Combine into pipe-separated row
    return (
        f"{prefix}| {dataset_col} | {phase_col} | {processed_col} | {matches_col} "
        f"| {rate_col} | {progress_col} | {eta_col} |"
    )


def print_progress_header(prefix: str = "") -> None:
    """Print the header for progress output.

    Prints a formatted table header with column names and separator
    lines. Should be called once before printing progress lines.
    """
    header = _progress_header_row()

    # Separator line spans header width (printed with the same prefix)
    sep_line = "-" * len(header)

    # Print header block with prefix applied to all lines
    print(prefix + sep_line)
    print(prefix + header)
    print(prefix + sep_line)


def _progress_header_row() -> str:
    """Return the Stage 1 progress table header row (no prefix)."""
    return (
        f"| {'Dataset':<14} | {'Phase':<11} | {'Processed':>12} | {'Matches':>10} "
        f"| {'Rate':>10} | {'Prog':>6} | {'ETA':>8} |"
    )


def format_progress_file_complete(
    dataset: str,
    phase: str,
    processed: int,
    matches: int,
    prefix: str = "",
) -> str:
    """Format a completion message that visually interrupts the Stage 1 table.

    The output is centered within the table width using '-' fill, matching the
    style used by Stage 2's Pass 1 completion lines.
    """
    table_width = len(_progress_header_row())
    msg = f" {dataset}/{phase}: {processed:,} processed, {matches:,} matches "
    return f"{prefix}{msg:-^{table_width}}"
