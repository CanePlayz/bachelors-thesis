"""Shared stats data models for the v2 pipeline.

This module defines the data classes and types used for
tracking and reporting statistics throughout the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DatasetStats:
    """Statistics for a single dataset.

    Tracks counts and metrics for one subreddit dataset
    (e.g., 'investing' or 'wallstreetbets').

    Attributes:
        dataset: Name of the dataset (subreddit name)
        total_submissions: Total number of submissions processed
        matched_submissions: Number of submissions with ticker mentions
        total_comments: Total number of comments processed
        matched_comments: Number of comments with ticker mentions
        unique_tickers_subs: Number of unique tickers in submissions
        unique_tickers_comments: Number of unique tickers in comments
        date_range: Tuple of (earliest_date, latest_date) or None
        top_tickers: List of (ticker, count) tuples for top 20
        runtime_seconds: Time taken to process this dataset
    """

    dataset: str
    total_submissions: int = 0
    matched_submissions: int = 0
    total_comments: int = 0
    matched_comments: int = 0
    unique_tickers_subs: int = 0
    unique_tickers_comments: int = 0
    date_range: Optional[Tuple[str, str]] = None
    top_tickers: List[Tuple[str, int]] = field(default_factory=list)
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary with all stats fields
        """
        return {
            "dataset": self.dataset,
            "total_submissions": self.total_submissions,
            "matched_submissions": self.matched_submissions,
            "total_comments": self.total_comments,
            "matched_comments": self.matched_comments,
            "unique_tickers_subs": self.unique_tickers_subs,
            "unique_tickers_comments": self.unique_tickers_comments,
            "date_range": self.date_range,
            "top_tickers": self.top_tickers,
            "runtime_seconds": self.runtime_seconds,
        }
