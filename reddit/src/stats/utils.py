"""Small stats utility helpers for v2 pipeline.

This module provides simple statistical functions used by
the pipeline for computing distribution metrics.
"""

from __future__ import annotations

from typing import List


def percentile(values: List[int], p: float) -> float:
    """Compute percentile using linear interpolation.

    Uses a standard linear interpolation method to estimate
    the value at percentile p in the sorted list of values.

    Args:
        values: List of integer values to compute percentile from
        p: Percentile to compute (0-100)

    Returns:
        The interpolated value at percentile p.
        Returns 0.0 for empty input.

    Example:
        >>> percentile([1, 2, 3, 4, 5], 50)
        3.0
        >>> percentile([1, 2, 3, 4, 5], 90)
        4.6
    """
    # Handle empty list
    if not values:
        return 0.0

    # Handle single value
    if len(values) == 1:
        return float(values[0])

    # Sort values
    vals = sorted(values)

    # Compute fractional index for percentile
    k = (p / 100.0) * (len(vals) - 1)

    # Get lower and upper indices
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)

    # Compute interpolation fraction
    frac = k - lo

    # Return linearly interpolated value
    return vals[lo] + (vals[hi] - vals[lo]) * frac
