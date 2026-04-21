"""LMDB-backed sentiment cache for storing model scores.

This module provides a persistent cache for sentiment scores, keyed by
text hash and optionally a target ticker. This avoids re-scoring the same
text multiple times across different pipeline runs.

Key features:
- SHA256 text hashing for deduplication
- Per-model LMDB databases (shared across pipelines)
- Support for targeted models (text + ticker combinations)
- Batch lookup and insert operations for efficiency

Cache location: CACHES_DIR/{model_name_sanitized}.lmdb
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lmdb
from data.config import CACHES_DIR


@dataclass
class CachedScore:
    """A single cached sentiment score.

    Attributes:
        score: Signed sentiment value (typically -1 to +1)
        label: Model's predicted label (e.g., "positive", "negative")
    """

    score: float
    label: str


def sanitize_model_name(model_name: str) -> str:
    """Convert a HuggingFace model name to a safe filename.

    Removes slashes and special characters, keeping only alphanumeric
    and underscores. This creates a valid filename for the cache.

    Args:
        model_name: HuggingFace model identifier (e.g., "cardiffnlp/twitter-roberta-base")

    Returns:
        Sanitized string safe for use as a filename
    """
    # Replace slashes with underscores first
    name = model_name.replace("/", "_")
    # Keep only alphanumeric, underscore, and dash
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


class SentimentCache:
    """LMDB-backed cache for sentiment scores.

    Each sentiment model gets its own LMDB directory. The cache supports
    both targeted (text + ticker) and non-targeted (text only) lookups.

    Key format:
        hash (TEXT) -- SHA256 of (text + target)

    Value format:
        "{score}\x00{label}" (UTF-8 encoded)
    """

    _DEFAULT_MAP_SIZE = 4 * 1024 * 1024 * 1024  # 4GB
    _MAP_GROWTH = 2 * 1024 * 1024 * 1024  # 2GB

    def __init__(self, model_name: str):
        """Initialize cache for a specific model.

        Creates the cache directory and database file if they don't exist.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name

        # Build path to cache directory
        safe_name = sanitize_model_name(model_name)
        self.cache_path = os.path.join(CACHES_DIR, f"{safe_name}.lmdb")

        # Ensure cache directory exists
        os.makedirs(CACHES_DIR, exist_ok=True)

        # Open LMDB environment
        os.makedirs(self.cache_path, exist_ok=True)
        self._map_size = self._DEFAULT_MAP_SIZE
        self._env = lmdb.open(
            self.cache_path,
            map_size=self._map_size,
            max_readers=126,
            metasync=False,
            sync=False,
            readonly=False,
        )

    def _grow_map_size(self) -> None:
        """Grow LMDB map size when full."""
        self._map_size += self._MAP_GROWTH
        self._env.set_mapsize(self._map_size)

    @staticmethod
    def _hash_key(text: str, target: Optional[str] = None) -> str:
        """Generate a unique hash key for a text (+ optional target).

        For non-targeted models, only the text is hashed.
        For targeted models, both text and target are combined.

        Args:
            text: The text to score
            target: Optional ticker/target for targeted models

        Returns:
            SHA256 hex digest string
        """
        # Combine text and target for targeted models
        key = text if target is None else f"{text}|||{target}"
        return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _make_value(score: float, label: str) -> bytes:
        return f"{score}\x00{label}".encode("utf-8")

    @staticmethod
    def _parse_value(value: bytes) -> CachedScore:
        parts = value.decode("utf-8").split("\x00")
        return CachedScore(score=float(parts[0]), label=parts[1])

    def get(self, text: str, target: Optional[str] = None) -> Optional[CachedScore]:
        """Look up a cached score for a text.

        Args:
            text: The text to look up
            target: Optional target ticker (for targeted models)

        Returns:
            CachedScore if found, None if not in cache
        """
        h = self._hash_key(text, target)
        with self._env.begin() as txn:
            value = txn.get(h.encode("utf-8"))
            if value is None:
                return None
            return self._parse_value(bytes(value))

    def get_batch(
        self, items: List[Tuple[str, Optional[str]]]
    ) -> Dict[Tuple[str, Optional[str]], CachedScore]:
        """Look up multiple items at once.

        More efficient than individual lookups for large batches.

        Args:
            items: List of (text, target) tuples to look up

        Returns:
            Dict mapping (text, target) -> CachedScore for found items
        """
        if not items:
            return {}

        # Build hash -> item mapping
        hash_to_item: Dict[str, Tuple[str, Optional[str]]] = {}
        for text, target in items:
            h = self._hash_key(text, target)
            hash_to_item[h] = (text, target)

        results: Dict[Tuple[str, Optional[str]], CachedScore] = {}
        with self._env.begin() as txn:
            for h, item in hash_to_item.items():
                value = txn.get(h.encode("utf-8"))
                if value:
                    results[item] = self._parse_value(bytes(value))

        return results

    def put(
        self, text: str, score: float, label: str, target: Optional[str] = None
    ) -> None:
        """Store a score in the cache.

        Uses INSERT OR REPLACE to handle re-scoring if model versions change.

        Args:
            text: The scored text
            score: Signed sentiment value
            label: Model's predicted label
            target: Optional target ticker (for targeted models)
        """
        h = self._hash_key(text, target)
        while True:
            try:
                with self._env.begin(write=True) as txn:
                    txn.put(h.encode("utf-8"), self._make_value(score, label))
                break
            except lmdb.MapFullError:
                self._grow_map_size()

    def put_batch(self, items: List[Tuple[str, float, str, Optional[str]]]) -> None:
        """Store multiple scores at once.

        More efficient than individual puts for large batches.

        Args:
            items: List of (text, score, label, target) tuples
        """
        if not items:
            return

        rows = [
            (self._hash_key(text, target), score, label)
            for text, score, label, target in items
        ]
        while True:
            try:
                with self._env.begin(write=True) as txn:
                    for h, score, label in rows:
                        txn.put(h.encode("utf-8"), self._make_value(score, label))
                break
            except lmdb.MapFullError:
                self._grow_map_size()

    def commit(self) -> None:
        """Commit any pending changes to disk."""
        self._env.sync()

    def close(self) -> None:
        """Close the database connection."""
        self.commit()
        self._env.close()

    def __enter__(self) -> "SentimentCache":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - commits and closes."""
        self.close()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with 'total_entries' count
        """
        with self._env.begin() as txn:
            stat = txn.stat()  # type: ignore[call-arg]
            return {"total_entries": int(stat["entries"])}
