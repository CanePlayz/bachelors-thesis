"""LMDB-backed score store for fast sentiment score lookups.

LMDB (Lightning Memory-Mapped Database) provides ~10x faster reads than SQLite
for key-value workloads due to memory-mapped I/O and B+ tree structure.

This module provides:
- LMDBScoreStore: Fast score storage and retrieval for Stage 2 scoring

Key format: "{text_hash}|{model_key}|{target or ''}" (text is SHA-256 hashed to fit LMDB's 511-byte key limit)
Value format: "{score}|{label}" (UTF-8 encoded)
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

import lmdb

# Initial DB size: 1GB (will auto-grow when needed)
# LMDB on Windows allocates the full map_size on disk, so start small
DEFAULT_MAP_SIZE = 1 * 1024 * 1024 * 1024
# Growth increment when DB is full
MAP_SIZE_INCREMENT = 1 * 1024 * 1024 * 1024


def _hash_text(text: str) -> str:
    """Hash text to a fixed-size string for use in LMDB keys.

    LMDB has a max key size of 511 bytes. Using SHA-256 (64 hex chars)
    ensures we stay well under this limit regardless of text length.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_key(text: str, model_key: str, target: Optional[str] = None) -> bytes:
    """Create a key for the LMDB store.

    Format: text_hash|model_key|target (target empty string if None)
    Text is hashed to ensure key fits within LMDB's 511-byte limit.
    """
    text_hash = _hash_text(text)
    target_str = target if target is not None else ""
    return f"{text_hash}\x00{model_key}\x00{target_str}".encode("utf-8")


def _parse_key(key: bytes) -> Tuple[str, str, Optional[str]]:
    """Parse a key back into components.

    Note: Returns (text_hash, model_key, target) - the text hash, not original text.
    """
    parts = key.decode("utf-8").split("\x00")
    text_hash, model_key, target_str = (
        parts[0],
        parts[1],
        parts[2] if len(parts) > 2 else "",
    )
    return text_hash, model_key, target_str if target_str else None


def _make_value(score: float, label: str) -> bytes:
    """Create a value for the LMDB store."""
    return f"{score}\x00{label}".encode("utf-8")


def _parse_value(value: bytes) -> Tuple[float, str]:
    """Parse a value back into score and label."""
    parts = value.decode("utf-8").split("\x00")
    return float(parts[0]), parts[1]


class LMDBScoreStore:
    """LMDB-backed storage for sentiment scores.

    Provides fast key-value storage for sentiment scores, optimized for:
    - Fast batch lookups (memory-mapped reads)
    - Efficient batch writes
    - Crash-safe transactions

    Usage:
        with LMDBScoreStore("/path/to/scores.lmdb") as store:
            store.add_non_targeted([("text", "model", 0.5, "positive")])
            scores = store.get_scores_for_texts(["text"])
    """

    def __init__(self, db_path: str, map_size: int = DEFAULT_MAP_SIZE):
        """Initialize LMDB store.

        Args:
            db_path: Path to LMDB directory (will be created)
            map_size: Maximum database size in bytes
        """
        self._db_path = db_path
        self._map_size = map_size
        self._env: Optional[lmdb.Environment] = None

    def __enter__(self) -> "LMDBScoreStore":
        # If directory exists but is empty/corrupt (no data.mdb), remove it
        # so LMDB can create a fresh database
        if os.path.isdir(self._db_path):
            data_file = os.path.join(self._db_path, "data.mdb")
            if not os.path.isfile(data_file):
                import shutil

                shutil.rmtree(self._db_path)
        os.makedirs(self._db_path, exist_ok=True)
        self._env = lmdb.open(
            self._db_path,
            map_size=self._map_size,
            # Allow multiple readers
            max_readers=126,
            # Don't sync on every write (faster, still crash-safe with transactions)
            metasync=False,
            sync=False,
            # Allow writes
            readonly=False,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._env:
            self._env.sync()
            self._env.close()
            self._env = None

    def sync(self) -> None:
        """Force sync to disk."""
        if self._env:
            self._env.sync()

    def count(self) -> int:
        """Return total number of score entries."""
        if not self._env:
            return 0
        with self._env.begin() as txn:
            stat = txn.stat(db=None)
            return int(stat["entries"])

    def count_for_model(self, model_key: str, targeted: bool = False) -> int:
        """Count scores for a specific model.

        Note: This is O(n) - iterates all keys. Use sparingly.
        """
        if not self._env:
            return 0
        count = 0
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                _, mk, target = _parse_key(bytes(key))  # type: ignore[arg-type]
                if mk == model_key:
                    if targeted and target:
                        count += 1
                    elif not targeted and not target:
                        count += 1
        return count

    def _grow_map_size(self) -> None:
        """Grow the map size by MAP_SIZE_INCREMENT when database is full."""
        if not self._env:
            return
        old_size = self._map_size
        self._map_size += MAP_SIZE_INCREMENT
        # Close and reopen with new size
        self._env.sync()
        self._env.close()
        self._env = lmdb.open(
            self._db_path,
            map_size=self._map_size,
            max_readers=126,
            metasync=False,
            sync=False,
            readonly=False,
        )
        """ print(
            f"[LMDB] Grew map size: {old_size / (1024**3):.1f}GB → {self._map_size / (1024**3):.1f}GB"
        ) """

    def add_non_targeted(self, items: List[Tuple[str, str, float, str]]) -> None:
        """Add non-targeted scores: [(text, model_key, score, label), ...]"""
        if not self._env or not items:
            return
        try:
            with self._env.begin(write=True) as txn:
                for text, model_key, score, label in items:
                    key = _make_key(text, model_key, None)
                    value = _make_value(score, label)
                    txn.put(key, value)
        except lmdb.MapFullError:
            self._grow_map_size()
            self.add_non_targeted(items)  # Retry after growing

    def add_targeted(self, items: List[Tuple[str, str, str, float, str]]) -> None:
        """Add targeted scores: [(text, model_key, target, score, label), ...]"""
        if not self._env or not items:
            return
        try:
            with self._env.begin(write=True) as txn:
                for text, model_key, target, score, label in items:
                    key = _make_key(text, model_key, target)
                    value = _make_value(score, label)
                    txn.put(key, value)
        except lmdb.MapFullError:
            self._grow_map_size()
            self.add_targeted(items)  # Retry after growing

    def get_score(
        self, text: str, model_key: str, target: Optional[str] = None
    ) -> Optional[Tuple[float, str]]:
        """Get a single score."""
        if not self._env:
            return None
        with self._env.begin() as txn:
            key = _make_key(text, model_key, target)
            value = txn.get(key)
            if value:
                return _parse_value(bytes(value))
        return None

    def get_scores_batch(
        self, lookups: List[Tuple[str, str, Optional[str]]]
    ) -> Dict[Tuple[str, str, Optional[str]], Tuple[float, str]]:
        """Get scores for multiple (text, model_key, target) tuples.

        Args:
            lookups: List of (text, model_key, target) tuples

        Returns:
            Dict mapping found tuples to (score, label)
        """
        if not self._env or not lookups:
            return {}
        results = {}
        with self._env.begin() as txn:
            for text, model_key, target in lookups:
                key = _make_key(text, model_key, target)
                value = txn.get(key)
                if value:
                    results[(text, model_key, target)] = _parse_value(bytes(value))
        return results

    def get_all_scores_for_text(
        self, text: str
    ) -> Dict[str, Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]]]:
        """Get all scores for a single text across all models.

        Returns:
            Dict mapping model_key to (score, label, targeted_dict)
            where targeted_dict maps target -> score
        """
        if not self._env:
            return {}

        # Hash the text to match the key format
        text_hash = _hash_text(text)
        # We need to scan for all keys starting with this text hash
        # LMDB doesn't have prefix scanning, so we use a cursor
        prefix = f"{text_hash}\x00".encode("utf-8")
        results: Dict[
            str, Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]]
        ] = {}

        with self._env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, value in cursor:
                    key = bytes(key)
                    if not key.startswith(prefix):
                        break
                    _, model_key, target = _parse_key(key)
                    score, label = _parse_value(value)

                    if model_key not in results:
                        results[model_key] = (None, None, None)

                    cur_score, cur_label, cur_targeted = results[model_key]

                    if target is None:
                        results[model_key] = (score, label, cur_targeted)
                    else:
                        if cur_targeted is None:
                            cur_targeted = {}
                        cur_targeted[target] = score
                        results[model_key] = (cur_score, cur_label, cur_targeted)

        return results

    def fetch_scores_for_texts(self, texts: List[str]) -> Dict[
        str,
        Dict[str, Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]]],
    ]:
        """Fetch all scores for a list of texts.

        This is the main lookup method for Pass 2 (applying scores).
        Optimized to use a single transaction and batch prefix lookups.

        Args:
            texts: List of text strings to look up

        Returns:
            Nested dict: text -> model_key -> (score, label, targeted_dict)
        """
        if not self._env or not texts:
            return {}

        # Pre-compute hashes and build lookup map
        hash_to_text: Dict[str, str] = {}
        for text in texts:
            text_hash = _hash_text(text)
            hash_to_text[text_hash] = text

        # Sort hashes for sequential cursor access (much faster)
        sorted_hashes = sorted(hash_to_text.keys())

        results: Dict[
            str,
            Dict[
                str, Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]]
            ],
        ] = {}

        # Single transaction for all lookups
        with self._env.begin() as txn:
            cursor = txn.cursor()

            for text_hash in sorted_hashes:
                text = hash_to_text[text_hash]
                prefix = f"{text_hash}\x00".encode("utf-8")

                if not cursor.set_range(prefix):
                    continue

                text_results: Dict[
                    str,
                    Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]],
                ] = {}

                for key, value in cursor:
                    key = bytes(key)
                    if not key.startswith(prefix):
                        break
                    _, model_key, target = _parse_key(key)
                    score, label = _parse_value(value)

                    if model_key not in text_results:
                        text_results[model_key] = (None, None, None)

                    cur_score, cur_label, cur_targeted = text_results[model_key]

                    if target is None:
                        text_results[model_key] = (score, label, cur_targeted)
                    else:
                        if cur_targeted is None:
                            cur_targeted = {}
                        cur_targeted[target] = score
                        text_results[model_key] = (cur_score, cur_label, cur_targeted)

                if text_results:
                    results[text] = text_results

        return results

    def has_score(
        self, text: str, model_key: str, target: Optional[str] = None
    ) -> bool:
        """Check if a score exists."""
        if not self._env:
            return False
        with self._env.begin() as txn:
            key = _make_key(text, model_key, target)
            return txn.get(key) is not None

    def get_scored_texts_for_model(self, model_key: str) -> Set[str]:
        """Get all texts that have non-targeted scores for a model.

        Note: O(n) scan - use sparingly.
        """
        if not self._env:
            return set()
        texts = set()
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                text, mk, target = _parse_key(bytes(key))  # type: ignore[arg-type]
                if mk == model_key and target is None:
                    texts.add(text)
        return texts

    def get_scored_pairs_for_model(self, model_key: str) -> Set[Tuple[str, str]]:
        """Get all (text, target) pairs that have targeted scores for a model.

        Note: O(n) scan - use sparingly.
        """
        if not self._env:
            return set()
        pairs = set()
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                text, mk, target = _parse_key(bytes(key))  # type: ignore[arg-type]
                if mk == model_key and target is not None:
                    pairs.add((text, target))
        return pairs

    def iter_all(self) -> Iterator[Tuple[str, str, Optional[str], float, str]]:
        """Iterate over all scores.

        Yields: (text, model_key, target, score, label)
        """
        if not self._env:
            return
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                text, model_key, target = _parse_key(key)
                score, label = _parse_value(value)
                yield text, model_key, target, score, label


def delete_lmdb_store(path: str) -> None:
    """Delete an LMDB store directory."""
    import shutil

    if os.path.exists(path):
        shutil.rmtree(path)
