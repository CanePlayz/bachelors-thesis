"""IO helpers for streaming Reddit NDJSON data inside .zst archives.

This module provides efficient streaming utilities for reading and writing
Reddit data stored as newline-delimited JSON (NDJSON) inside Zstandard (.zst)
compressed archives. The streaming approach ensures memory-efficient processing
of multi-gigabyte datasets.

Key functions:
- iter_zst_ndjson(): Stream JSON objects from a .zst file
- iter_zst_ndjson_with_progress(): Same as above, with progress tracking
- write_jsonl_zst(): Write JSON objects to a compressed .zst file

Key classes:
- StreamProgress: Track bytes read, items read, and estimate progress
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

# orjson is a fast JSON library written in Rust
# Much faster than stdlib json for large-scale parsing
import orjson

# zstandard provides Python bindings to the Zstd compression library
# Used by Reddit data dumps for efficient compression
import zstandard as zstd

# =============================================================================
# JSON serialization helpers
# =============================================================================


def _loads(s: bytes) -> dict:
    """Parse a JSON bytes string into a Python dict.

    Uses orjson for fast parsing (3-10x faster than stdlib json).

    Args:
        s: UTF-8 encoded JSON bytes

    Returns:
        Parsed dict object
    """
    return orjson.loads(s)


def _dumps(obj: dict) -> bytes:
    """Serialize a dict to JSON bytes with trailing newline.

    Uses orjson with OPT_APPEND_NEWLINE for NDJSON format.

    Args:
        obj: Dict to serialize

    Returns:
        UTF-8 encoded JSON bytes with trailing newline
    """
    return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE)


# =============================================================================
# Streaming configuration
# =============================================================================

# Chunk size for streaming reads (64 KB)
# This is tuned for a balance between:
# - Memory efficiency (not loading too much at once)
# - IO efficiency (not making too many read syscalls)
# - Decompression efficiency (giving zstd enough data to work with)
_CHUNK_SIZE = 64 * 1024


# =============================================================================
# Progress tracking
# =============================================================================


@dataclass
class StreamProgress:
    """Track progress of a streaming read operation.

    This dataclass maintains state about how much of a file has been
    read and how many items have been parsed. It can be passed to
    iter_zst_ndjson_with_progress() to enable real-time progress updates.

    Attributes:
        bytes_read: Number of compressed bytes read from disk so far
        total_bytes: Total size of the compressed file in bytes
        items_read: Number of JSON objects successfully parsed so far
    """

    bytes_read: int = 0
    total_bytes: int = 0
    items_read: int = 0

    @property
    def progress_pct(self) -> float:
        """Return percentage progress (0-100) based on bytes read.

        Note: This is based on compressed bytes, not decompressed size.
        Progress may appear non-linear due to varying compression ratios.

        Returns:
            Float between 0.0 and 100.0 representing completion percentage
        """
        # Guard against division by zero
        if self.total_bytes <= 0:
            return 0.0

        # Cap at 100% in case of rounding issues
        return min(100.0, 100.0 * self.bytes_read / self.total_bytes)

    def estimate_total_items(self) -> int:
        """Estimate total items based on current progress.

        Uses linear extrapolation: if we've read X% of bytes and
        found N items, we estimate total items as N / (X/100).

        Returns:
            Estimated total number of items in the file
        """
        # Need some data to make an estimate
        if self.bytes_read <= 0 or self.items_read <= 0:
            return 0

        # Linear extrapolation based on bytes read
        return int(self.items_read * self.total_bytes / self.bytes_read)


# =============================================================================
# Streaming readers
# =============================================================================


def iter_zst_ndjson(path: str) -> Iterator[dict]:
    """Yield JSON objects from a .zst file containing one JSON document per line.

    This is the core streaming reader for Reddit data dumps. It:
    1. Opens the .zst file in binary mode
    2. Creates a Zstandard decompressor with large window size
    3. Reads chunks of decompressed data
    4. Splits on newlines and parses each line as JSON
    5. Yields each parsed dict, skipping malformed lines

    Args:
        path: Path to the .zst file

    Yields:
        Dict objects parsed from each JSON line

    Example:
        for post in iter_zst_ndjson("submissions.zst"):
            print(post["title"])
    """
    # Open file in binary mode for zstd decompression
    with open(path, "rb") as f:
        # Create decompressor with max window size for large archives
        # 2**31 (2GB) handles the largest Reddit dumps
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)

        # Create streaming reader context
        with dctx.stream_reader(f) as reader:
            # Buffer to accumulate partial lines across chunks
            buff = b""

            # Read chunks until EOF
            while True:
                # Read next chunk of decompressed data
                chunk = reader.read(_CHUNK_SIZE)

                # Empty chunk means we've reached EOF
                if not chunk:
                    break

                # Append chunk to buffer
                buff += chunk

                # Extract and yield complete lines from buffer
                while True:
                    # Find next newline character
                    nl = buff.find(b"\n")

                    # No newline found - need more data
                    if nl == -1:
                        break

                    # Extract line up to newline
                    line = buff[:nl]

                    # Remove line from buffer (including newline)
                    buff = buff[nl + 1 :]

                    # Skip empty or whitespace-only lines
                    if not line or line.isspace():
                        continue

                    # Try to parse JSON
                    try:
                        yield _loads(line)
                    except (ValueError, TypeError):
                        # Skip malformed JSON lines (rare but possible)
                        continue

            # Handle any leftover data in buffer (file may not end with newline)
            line = buff.strip()
            if line:
                try:
                    yield _loads(line)
                except (ValueError, TypeError):
                    # Skip malformed final line
                    return


def iter_zst_ndjson_with_progress(
    path: str,
    progress: Optional[StreamProgress] = None,
) -> Iterator[dict]:
    """Yield JSON objects from a .zst file, updating progress tracker.

    Same as iter_zst_ndjson() but also updates a StreamProgress object
    to enable real-time progress reporting. The progress is based on
    compressed bytes read from disk.

    Args:
        path: Path to .zst file
        progress: Optional StreamProgress object to update with bytes read.
                  If None, behaves identically to iter_zst_ndjson().

    Yields:
        Dict objects parsed from each JSON line

    Example:
        progress = StreamProgress()
        for post in iter_zst_ndjson_with_progress("data.zst", progress):
            if progress.items_read % 10000 == 0:
                print(f"Progress: {progress.progress_pct:.1f}%")
    """
    # Get file size for progress percentage calculation
    file_size = os.path.getsize(path)

    # Initialize progress tracker if provided
    if progress is not None:
        progress.total_bytes = file_size
        progress.bytes_read = 0
        progress.items_read = 0

    # Open file in binary mode
    with open(path, "rb") as f:
        # Create decompressor with large window size
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)

        # Create streaming reader
        with dctx.stream_reader(f) as reader:
            # Buffer for partial lines
            buff = b""

            # Counter for items read
            items_count = 0

            # Read chunks until EOF
            while True:
                # Read next chunk
                chunk = reader.read(_CHUNK_SIZE)

                # Empty chunk = EOF
                if not chunk:
                    break

                # Append to buffer
                buff += chunk

                # Update progress with current file position
                # f.tell() gives compressed bytes read
                if progress is not None:
                    progress.bytes_read = f.tell()

                # Extract complete lines
                while True:
                    # Find newline
                    nl = buff.find(b"\n")

                    # No newline - need more data
                    if nl == -1:
                        break

                    # Extract line
                    line = buff[:nl]
                    buff = buff[nl + 1 :]

                    # Skip empty lines
                    if not line or line.isspace():
                        continue

                    # Parse JSON
                    try:
                        obj = _loads(line)

                        # Increment item counter
                        items_count += 1

                        # Update progress
                        if progress is not None:
                            progress.items_read = items_count

                        yield obj
                    except (ValueError, TypeError):
                        # Skip malformed lines
                        continue

            # Final progress update
            if progress is not None:
                progress.bytes_read = f.tell()

            # Handle leftover buffer
            line = buff.strip()
            if line:
                try:
                    obj = _loads(line)
                    items_count += 1
                    if progress is not None:
                        progress.items_read = items_count
                    yield obj
                except (ValueError, TypeError):
                    return


# =============================================================================
# Streaming writer
# =============================================================================


def write_jsonl_zst(path: str, rows: Iterable[dict], level: int = 10) -> None:
    """Write iterable of dict rows as newline-delimited JSON inside a .zst archive.

    Creates the output directory if it doesn't exist. Uses streaming
    compression to handle large outputs without loading everything into memory.

    Args:
        path: Output path for the .zst file
        rows: Iterable of dicts to serialize (can be a generator)
        level: Zstandard compression level (1-22). Default 10 provides
               good balance of speed and compression ratio. Higher values
               compress better but are slower.

    Example:
        def generate_records():
            for i in range(1000000):
                yield {"id": i, "data": "..."}

        write_jsonl_zst("output.jsonl.zst", generate_records())
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create compressor with specified level
    # Level 10 is a good default (level 1 = fast, level 22 = max compression)
    cctx = zstd.ZstdCompressor(level=level)

    # Open file in binary write mode
    with open(path, "wb") as f:
        # Create streaming writer context
        with cctx.stream_writer(f) as writer:
            # Write each row as JSON line
            for row in rows:
                # _dumps adds trailing newline
                writer.write(_dumps(row))
