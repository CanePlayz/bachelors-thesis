"""Sentiment analysis module for v2 pipeline.

This module provides sentiment scoring and aggregation capabilities:
- Stage 2: Score mentions with multiple sentiment models (GPU-accelerated)
- Stage 3: Aggregate sentiment by day, stock, and model

Key components:
- cache.py: SQLite-backed sentiment score cache
- model_loader.py: HuggingFace model loading and batch scoring
- scorer.py: Stage 2 implementation (score all mentions)
- aggregate.py: Stage 3 implementation (daily aggregation)

Usage:
    from sentiment import run_stage2, run_stage3

    # Score all mentions with all models
    run_stage2(datasets=["investing", "wallstreetbets"])

    # Aggregate to daily sentiment per stock
    run_stage3()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path so `from common.io_utils import ...` works.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress some Transformers advisory messages (partial mitigation).
# The main suppression is done via FD-level stdout/stderr redirect in model_loader.py.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from .aggregate import run_stage3
from .scorer import run_stage2

__all__ = ["run_stage2", "run_stage3"]
