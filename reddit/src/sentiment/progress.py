"""Progress formatting utilities for Stage 2 sentiment scoring.

Provides fixed-width table output for scoring progress, consistent
with Stage 1 style but adapted for Stage 2's cache-check + scoring
workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from helpers.logging_utils import format_duration


def format_stage2_line(
    checked: int,
    total: int,
    scored: int,
    cached: int,
    rate: float,
    progress_pct: float,
    eta_seconds: float,
    prefix: str = "  ",
) -> str:
    """Format a progress row for Stage 2 sentiment scoring.

    Args:
        checked: Number of items inspected so far
        total: Total items to process
        scored: Number of items scored (not from cache)
        cached: Number of cache hits
        rate: Processing rate in items/second
        progress_pct: Completion percentage (0-100)
        eta_seconds: Estimated seconds remaining
        prefix: Left padding for indentation

    Returns:
        Formatted table row
    """
    checked_col = f"{checked:>12,}"
    total_col = f"{total:>12,}"
    scored_col = f"{scored:>10,}"
    cached_col = f"{cached:>10,}"
    rate_col = f"{rate:>8,.0f}/s"
    progress_col = f"{progress_pct:>5.1f}%"
    eta_col = f"{format_duration(eta_seconds):>8}"

    return (
        f"{prefix}| {checked_col} | {total_col} | {scored_col} | {cached_col} "
        f"| {rate_col} | {progress_col} | {eta_col} |"
    )


def print_stage2_header(prefix: str = "  ") -> None:
    """Print header for Stage 2 scoring progress table."""
    header_body = (
        f"| {'Checked':>12} | {'Total':>12} | {'Scored':>10} | {'Cache hits':>10} "
        f"| {'Rate':>10} | {'Prog':>6} | {'ETA':>8} |"
    )
    sep_line = "-" * len(header_body)
    print(prefix + sep_line)
    print(prefix + header_body)
    print(prefix + sep_line)


def format_scoring_line(
    phase: str,
    processed: int,
    total: int,
    rate: float,
    progress_pct: float,
    eta_seconds: float,
    prefix: str = "",
) -> str:
    """Format a progress row for in-memory scoring (separate cache/score passes).

    Args:
        phase: Current phase name (e.g., "non-targeted", "targeted")
        processed: Number of items processed
        total: Total items to process
        rate: Processing rate in items/second
        progress_pct: Completion percentage (0-100)
        eta_seconds: Estimated seconds remaining
        prefix: Left padding

    Returns:
        Formatted table row
    """
    phase_col = f"{phase:<14}"[:14]
    processed_col = f"{processed:>10,}"
    total_col = f"{total:>10,}"
    rate_col = f"{rate:>8,.0f}/s"
    progress_col = f"{progress_pct:>5.1f}%"
    eta_col = f"{format_duration(eta_seconds):>8}"

    return (
        f"{prefix}| {phase_col} | {processed_col} | {total_col} "
        f"| {rate_col} | {progress_col} | {eta_col} |"
    )


def print_scoring_header(prefix: str = "") -> None:
    """Print header for in-memory scoring progress table."""
    header = (
        f"| {'Phase':<14} | {'Scored':>10} | {'Total':>10} "
        f"| {'Rate':>10} | {'Prog':>6} | {'ETA':>8} |"
    )
    sep_line = "-" * len(header)
    print(prefix + sep_line)
    print(prefix + header)
    print(prefix + sep_line)


def format_stage2_line_inmem(
    checked: int,
    total: int,
    scored: int,
    rate: float,
    progress_pct: float,
    eta_seconds: float,
    prefix: str = "  ",
) -> str:
    """Format a progress row for Stage 2 scoring in in-memory mode (no cache column)."""
    checked_col = f"{checked:>12,}"
    total_col = f"{total:>12,}"
    scored_col = f"{scored:>10,}"
    rate_col = f"{rate:>8,.0f}/s"
    progress_col = f"{progress_pct:>5.1f}%"
    eta_col = f"{format_duration(eta_seconds):>8}"

    return (
        f"{prefix}| {checked_col} | {total_col} | {scored_col} "
        f"| {rate_col} | {progress_col} | {eta_col} |"
    )


def print_stage2_header_inmem(prefix: str = "  ") -> None:
    """Print header for Stage 2 scoring progress table in in-memory mode."""
    header_body = (
        f"| {'Checked':>12} | {'Total':>12} | {'Scored':>10} "
        f"| {'Rate':>10} | {'Prog':>6} | {'ETA':>8} |"
    )
    sep_line = "-" * len(header_body)
    print(prefix + sep_line)
    print(prefix + header_body)
    print(prefix + sep_line)


def format_pass1_line(
    mentions: int,
    unique_texts: int,
    target_pairs: int,
    rate: float,
    progress_pct: float,
    eta_seconds: float,
    prefix: str = "  ",
) -> str:
    """Format a progress row for Pass 1 (collect unique texts).

    Columns mirror the Stage 1 table style for consistency.

    Args:
        mentions: Mentions scanned so far
        unique_texts: Unique texts observed
        target_pairs: (text, target) pairs observed
        rate: Mentions processed per second
        progress_pct: Completion percentage (0-100)
        eta_seconds: Estimated seconds remaining
        prefix: Left padding

    Returns:
        Formatted table row
    """

    mentions_col = f"{mentions:>12,}"
    uniq_col = f"{unique_texts:>12,}"
    pairs_col = f"{target_pairs:>12,}"
    rate_col = f"{rate:>8,.0f}/s"
    progress_col = f"{progress_pct:>5.1f}%"
    eta_col = f"{format_duration(eta_seconds):>8}"

    return (
        f"{prefix}| {mentions_col} | {uniq_col} | {pairs_col} "
        f"| {rate_col} | {progress_col} | {eta_col} |"
    )


def print_pass1_header(prefix: str = "  ") -> None:
    """Print header for Pass 1 scoring progress table."""
    header = (
        f"| {'Texts':>12} | {'Scored':>12} | {'Cache Hits':>12} "
        f"| {'Rate':>10} | {'Prog':>6} | {'ETA':>8} |"
    )
    sep_line = "-" * len(header)
    print(prefix + sep_line)
    print(prefix + header)
    print(prefix + sep_line)


def format_pass1_file_complete(
    dataset: str, file_type: str, count: int, prefix: str = "  "
) -> str:
    """Format a file completion message that fits within the table.

    Creates a centered message within a table-width row using the
    same separator character as the table borders.

    Args:
        dataset: Name of the dataset
        file_type: Type of file (submission/comment)
        count: Number of mentions processed
        prefix: Left padding

    Returns:
        Formatted completion row
    """
    # Table width matches the header (79 chars without prefix)
    table_width = 79
    msg = f" {dataset}/{file_type}: {count:,} mentions "
    # Center the message within dashes
    return f"{prefix}{msg:-^{table_width}}"


def format_pass2_line(
    total_records: int,
    dataset: str,
    file_type: str,
    file_records: int,
    rate: float,
    progress_pct: float,
    eta_seconds: float,
    prefix: str = "  ",
) -> str:
    """Format a progress row for Pass 2 (apply scores and write output).

    Args:
        total_records: Total records written so far
        dataset: Current dataset name
        file_type: Current file type (submission/comment)
        file_records: Records written in current file
        rate: Records per second
        progress_pct: Completion percentage (0-100)
        eta_seconds: Estimated seconds remaining
        prefix: Left padding

    Returns:
        Formatted table row
    """
    total_col = f"{total_records:>10,}"
    ds_col = f"{dataset[:12]:<12}"
    type_col = f"{file_type[:10]:<10}"
    file_col = f"{file_records:>10,}"
    rate_col = f"{rate:>8,.0f}/s"
    progress_col = f"{progress_pct:>5.1f}%"
    eta_col = f"{format_duration(eta_seconds):>8}"

    return (
        f"{prefix}| {total_col} | {ds_col} | {type_col} | {file_col} "
        f"| {rate_col} | {progress_col} | {eta_col} |"
    )


def print_pass2_header(prefix: str = "  ") -> None:
    """Print header for Pass 2 apply progress table."""
    header = (
        f"| {'Total':>10} | {'Dataset':<12} | {'Type':<10} | {'In File':>10} "
        f"| {'Rate':>10} | {'Prog':>6} | {'ETA':>8} |"
    )
    sep_line = "-" * len(header)
    print(prefix + sep_line)
    print(prefix + header)
    print(prefix + sep_line)


@dataclass
class ScoringProgress:
    """Track progress during sentiment scoring."""

    total_items: int = 0
    processed_items: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def progress_pct(self) -> float:
        if self.total_items <= 0:
            return 0.0
        return min(100.0, 100.0 * self.processed_items / self.total_items)

    @property
    def eta_seconds(self) -> float:
        if self.processed_items <= 0:
            return 0.0
        rate = self.processed_items / self.elapsed
        remaining = self.total_items - self.processed_items
        return remaining / rate if rate > 0 else 0.0

    @property
    def items_per_sec(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.processed_items / self.elapsed
