"""Unified pipeline state management with fingerprint-based invalidation.

This module provides a single, consistent system for tracking pipeline state
and detecting when stages need to be re-run. All stages use the same logic.

State Structure
---------------
The pipeline state is stored in `pipeline_state.json` with this structure:

    {
      "stage1": {
        "input_fp": "abc123",     // fingerprint of raw .zst dataset files
        "output_fp": "def456"     // fingerprint of s1_*.jsonl.zst files
      },
      "stage2": {
        "input_fp": "def456",     // must match stage1.output_fp
        "output_fp": "ghi789",    // fingerprint of s2_*.jsonl.zst files
        "lowmem": {               // only present in low-memory mode
          "pass1": {
            "input_fp": "def456", // same as stage2.input_fp
            "output_fp": "jkl012" // fingerprint of scores.lmdb/
          }
        }
      },
      "stage3": {
        "input_fp": "ghi789",     // must match stage2.output_fp
        "output_fp": "pqr678"     // fingerprint of daily_sentiment.csv
      }
    }

Validation Rules
----------------
A stage (or pass) is considered VALID and can be reused if ALL of these hold:

    1. output_fp is not None
       → Proves the stage completed (fingerprint only saved at end)

    2. input_fp == previous_stage.output_fp
       → Proves the stage used the correct inputs (not stale)

    3. output_fp == current_fingerprint(actual_outputs)
       → Proves outputs haven't been modified/deleted since completion

If any condition fails, the stage must be rebuilt.

Low-Memory Mode (Stage 2)
-------------------------
In low-memory mode, Stage 2 is split into:
    - Pass 1: Score mentions, cache to LMDB (scores.lmdb/)
    - Pass 2: Apply scores and write Stage 2 output files

Pass 2 doesn't need separate fingerprint tracking because:
    - If Stage 2 output is invalid → check Pass 1
    - If Pass 1 invalid → redo Pass 1 and Pass 2
    - If Pass 1 valid → only Pass 2 needs redo

The Stage 2 output fingerprint (output_fp) already serves as the
"Pass 2 complete" indicator.

Fingerprint Computation
-----------------------
Fingerprints are computed from file metadata (not content, for speed):
- File count
- Total size in bytes
- Latest modification time

This catches: file additions, deletions, modifications, and re-creations.

API Overview
------------
The module provides three namespaced classes for organization:

    Fingerprints: Compute fingerprints for various file types
        .raw_datasets(dir)      - Raw .zst input files
        .stage1_out(dir, ds)    - Stage 1 output files
        .stage2_out(dir, ds)    - Stage 2 output files
        .stage3_out(dir)        - Stage 3 output file
        .lmdb_store(path)       - LMDB database directory

    Check: Validate stage/pass outputs
        .stage1(state, datasets_dir, stage1_dir)
        .stage2(state, stage2_dir, datasets)
        .stage3(state, stage2_dir, stage3_dir)
        .lowmem_pass1(state, lmdb_path)

    Mark: Record stage/pass start/completion
        .stage1_started(state, datasets_dir)
        .stage1_complete(state, stage1_dir)
        .stage2_started(state, stage1_dir, datasets)
        .stage2_complete(state, stage2_dir, datasets)
        .stage3_started(state, stage2_dir)
        .stage3_complete(state, stage3_dir)
        .lowmem_pass1_started(state, stage1_dir, datasets)
        .lowmem_pass1_complete(state, lmdb_path)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import orjson

STATE_FILENAME = "pipeline_state.json"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StageState:
    """State of a single stage or pass.

    Attributes:
        input_fp: Fingerprint of inputs when stage started
        output_fp: Fingerprint of outputs when stage completed (None if incomplete)
    """

    input_fp: Optional[str] = None
    output_fp: Optional[str] = None


@dataclass
class LowMemState:
    """State of low-memory mode Pass 1 within Stage 2.

    In low-memory mode, Stage 2 is split into:
        - Pass 1: Score mentions, store in LMDB cache (scores.lmdb/)
        - Pass 2: Apply scores and write output (tracked by stage2.output_fp)

    Pass 2 doesn't need separate fingerprinting because the Stage 2
    output fingerprint already indicates Pass 2 completion.
    """

    pass1: StageState = field(default_factory=StageState)


@dataclass
class Stage2State:
    """State of Stage 2, including optional low-memory pass tracking.

    Attributes:
        input_fp: Fingerprint of Stage 1 outputs (inputs to Stage 2)
        output_fp: Fingerprint of Stage 2 outputs (s2_*.jsonl.zst files)
        lowmem: Optional state for low-memory mode passes
    """

    input_fp: Optional[str] = None
    output_fp: Optional[str] = None
    lowmem: Optional[LowMemState] = None


@dataclass
class PipelineState:
    """Complete pipeline state tracking all stages."""

    stage1: StageState = field(default_factory=StageState)
    stage2: Stage2State = field(default_factory=Stage2State)
    stage3: StageState = field(default_factory=StageState)


@dataclass
class ValidationResult:
    """Result of validating a stage or pass.

    Attributes:
        valid: True if the stage outputs can be reused
        reason: Human-readable explanation if invalid (None if valid)
    """

    valid: bool
    reason: Optional[str] = None


# =============================================================================
# Fingerprint Computation
# =============================================================================


def _compute_fingerprint(paths: List[str]) -> Optional[str]:
    """Compute a fingerprint from a list of file paths.

    The fingerprint captures:
    - Number of files that exist
    - Total size in bytes
    - Latest modification time

    Args:
        paths: List of file paths to include in fingerprint

    Returns:
        16-char hex digest string, or None if no files exist
    """
    stats = []
    for path in paths:
        if os.path.exists(path):
            st = os.stat(path)
            stats.append((st.st_size, st.st_mtime))

    if not stats:
        return None

    total_size = sum(s[0] for s in stats)
    latest_mtime = max(s[1] for s in stats)
    data = f"files={len(stats)}|size={total_size}|mtime={latest_mtime:.6f}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _fingerprint_stage_outputs(
    base_dir: str, datasets: Optional[List[str]], prefix: str
) -> Optional[str]:
    """Compute fingerprint of stage output files.

    Args:
        base_dir: Path to stage output directory (stage1_mentions or stage2_sentiment)
        datasets: List of dataset names to include, or None for all
        prefix: File prefix ("s1" or "s2")

    Returns:
        Fingerprint string, or None if no matching files
    """
    if not os.path.isdir(base_dir):
        return None

    paths = []
    subdirs = (
        datasets
        if datasets
        else [d for d in os.listdir(base_dir) if not d.startswith("_")]
    )

    for ds in subdirs:
        ds_dir = os.path.join(base_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for kind in ["submission", "comment"]:
            fpath = os.path.join(ds_dir, f"{prefix}_{kind}_mentions.jsonl.zst")
            if os.path.exists(fpath):
                paths.append(fpath)

    return _compute_fingerprint(sorted(paths))


class Fingerprints:
    """Namespace for fingerprint computation functions.

    All methods are static and compute fingerprints for different file types
    used throughout the pipeline.
    """

    @staticmethod
    def raw_datasets(datasets_dir: str) -> Optional[str]:
        """Compute fingerprint of raw .zst dataset files (Stage 1 inputs).

        Args:
            datasets_dir: Path to directory containing raw .zst files

        Returns:
            Fingerprint string, or None if no .zst files found
        """
        if not os.path.isdir(datasets_dir):
            return None
        paths = [
            os.path.join(datasets_dir, f)
            for f in os.listdir(datasets_dir)
            if f.endswith(".zst")
        ]
        return _compute_fingerprint(sorted(paths))

    @staticmethod
    def stage1_out(
        stage1_dir: str, datasets: Optional[List[str]] = None
    ) -> Optional[str]:
        """Compute fingerprint of Stage 1 outputs (Stage 2 inputs).

        Args:
            stage1_dir: Path to stage1_mentions directory
            datasets: Optional list of datasets to include. If None, includes all.

        Returns:
            Fingerprint string, or None if no output files found
        """
        return _fingerprint_stage_outputs(stage1_dir, datasets, "s1")

    @staticmethod
    def stage2_out(
        stage2_dir: str, datasets: Optional[List[str]] = None
    ) -> Optional[str]:
        """Compute fingerprint of Stage 2 outputs (Stage 3 inputs).

        Args:
            stage2_dir: Path to stage2_sentiment directory
            datasets: Optional list of datasets to include. If None, includes all.

        Returns:
            Fingerprint string, or None if no output files found
        """
        return _fingerprint_stage_outputs(stage2_dir, datasets, "s2")

    @staticmethod
    def stage3_out(stage3_dir: str) -> Optional[str]:
        """Compute fingerprint of Stage 3 output (daily_sentiment.csv).

        Args:
            stage3_dir: Path to stage3_aggregated directory

        Returns:
            Fingerprint string, or None if output file doesn't exist
        """
        return _compute_fingerprint([os.path.join(stage3_dir, "daily_sentiment.csv")])

    @staticmethod
    def lmdb_store(lmdb_path: str) -> Optional[str]:
        """Compute fingerprint of an LMDB store directory.

        LMDB stores consist of data.mdb and lock.mdb files.
        We only fingerprint data.mdb (the actual data file).

        Args:
            lmdb_path: Path to LMDB directory (e.g., scores.lmdb/)

        Returns:
            Fingerprint string, or None if store doesn't exist
        """
        data_file = os.path.join(lmdb_path, "data.mdb")
        return _compute_fingerprint([data_file])


# =============================================================================
# State Persistence
# =============================================================================


def load_state(base_dir: str) -> PipelineState:
    """Load pipeline state from disk.

    Args:
        base_dir: Directory containing pipeline_state.json

    Returns:
        PipelineState object (empty state if file doesn't exist or is invalid)
    """
    path = os.path.join(base_dir, STATE_FILENAME)
    if not os.path.exists(path):
        return PipelineState()

    try:
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        return _deserialize_state(data)
    except Exception:
        return PipelineState()


def save_state(base_dir: str, state: PipelineState) -> None:
    """Save pipeline state to disk.

    Args:
        base_dir: Directory to save pipeline_state.json
        state: PipelineState object to save
    """
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, STATE_FILENAME)
    with open(path, "wb") as f:
        f.write(orjson.dumps(_serialize_state(state), option=orjson.OPT_INDENT_2))


def clear_state(base_dir: str) -> None:
    """Clear pipeline state by removing the state file.

    Args:
        base_dir: Directory containing pipeline_state.json
    """
    path = os.path.join(base_dir, STATE_FILENAME)
    if os.path.exists(path):
        os.remove(path)


def _serialize_state(state: PipelineState) -> Dict[str, Any]:
    """Convert PipelineState to JSON-serializable dict."""
    result: Dict[str, Any] = {
        "stage1": asdict(state.stage1),
        "stage2": {
            "input_fp": state.stage2.input_fp,
            "output_fp": state.stage2.output_fp,
        },
        "stage3": asdict(state.stage3),
    }
    if state.stage2.lowmem is not None:
        result["stage2"]["lowmem"] = {
            "pass1": asdict(state.stage2.lowmem.pass1),
        }
    return result


def _deserialize_state(data: Dict[str, Any]) -> PipelineState:
    """Convert JSON dict to PipelineState."""
    state = PipelineState()

    if s1 := data.get("stage1"):
        state.stage1 = StageState(s1.get("input_fp"), s1.get("output_fp"))

    if s2 := data.get("stage2"):
        state.stage2 = Stage2State(s2.get("input_fp"), s2.get("output_fp"))
        if lm := s2.get("lowmem"):
            # Note: pass2 is ignored if present in legacy state files
            state.stage2.lowmem = LowMemState(
                pass1=StageState(
                    lm.get("pass1", {}).get("input_fp"),
                    lm.get("pass1", {}).get("output_fp"),
                ),
            )

    if s3 := data.get("stage3"):
        state.stage3 = StageState(s3.get("input_fp"), s3.get("output_fp"))

    return state


# =============================================================================
# Validation
# =============================================================================


def _validate(
    saved_input_fp: Optional[str],
    saved_output_fp: Optional[str],
    expected_input_fp: Optional[str],
    current_output_fp: Optional[str],
) -> ValidationResult:
    """Core validation logic for any stage or pass.

    A stage is VALID if ALL three conditions hold:

        1. output_fp is not None
           → Proves the stage completed (fingerprint only saved at end)

        2. input_fp == expected_input_fp (previous stage's output)
           → Proves the stage used the correct inputs (not stale)

        3. output_fp == current_fingerprint(actual_outputs)
           → Proves outputs haven't been modified/deleted since completion

    Args:
        saved_input_fp: input_fp from saved state
        saved_output_fp: output_fp from saved state
        expected_input_fp: What input_fp should be (previous stage's output_fp)
        current_output_fp: Current fingerprint of actual output files

    Returns:
        ValidationResult with valid and reason
    """
    if saved_output_fp is None:
        return ValidationResult(False, "no completion record (output_fp is None)")
    if saved_input_fp != expected_input_fp:
        return ValidationResult(
            False, f"input mismatch ({saved_input_fp} != {expected_input_fp})"
        )
    if saved_output_fp != current_output_fp:
        return ValidationResult(
            False, f"output changed ({saved_output_fp} != {current_output_fp})"
        )
    return ValidationResult(True)


class Check:
    """Namespace for stage/pass validation functions.

    All methods are static and return ValidationResult indicating whether
    the stage's outputs are valid and can be reused.
    """

    @staticmethod
    def stage1(
        state: PipelineState, datasets_dir: str, stage1_dir: str
    ) -> ValidationResult:
        """Check if Stage 1 outputs are valid and can be reused.

        Args:
            state: Current pipeline state
            datasets_dir: Path to raw datasets directory
            stage1_dir: Path to stage1_mentions directory

        Returns:
            ValidationResult indicating validity
        """
        return _validate(
            state.stage1.input_fp,
            state.stage1.output_fp,
            Fingerprints.raw_datasets(datasets_dir),
            Fingerprints.stage1_out(stage1_dir),
        )

    @staticmethod
    def stage2(
        state: PipelineState, stage2_dir: str, datasets: Optional[List[str]] = None
    ) -> ValidationResult:
        """Check if Stage 2 outputs are valid and can be reused.

        Args:
            state: Current pipeline state
            stage2_dir: Path to stage2_sentiment directory
            datasets: Optional list of datasets to check

        Returns:
            ValidationResult indicating validity
        """
        return _validate(
            state.stage2.input_fp,
            state.stage2.output_fp,
            state.stage1.output_fp,
            Fingerprints.stage2_out(stage2_dir, datasets),
        )

    @staticmethod
    def stage3(
        state: PipelineState, stage2_dir: str, stage3_dir: str
    ) -> ValidationResult:
        """Check if Stage 3 outputs are valid and can be reused.

        Args:
            state: Current pipeline state
            stage2_dir: Path to stage2_sentiment directory
            stage3_dir: Path to stage3_aggregated directory

        Returns:
            ValidationResult indicating validity
        """
        return _validate(
            state.stage3.input_fp,
            state.stage3.output_fp,
            state.stage2.output_fp,
            Fingerprints.stage3_out(stage3_dir),
        )

    @staticmethod
    def lowmem_pass1(state: PipelineState, lmdb_path: str) -> ValidationResult:
        """Check if Pass 1 (scoring to LMDB) is valid.

        Args:
            state: Current pipeline state
            lmdb_path: Path to scores.lmdb directory

        Returns:
            ValidationResult indicating validity
        """
        if not state.stage2.lowmem:
            return ValidationResult(False, "no low-memory state recorded")
        return _validate(
            state.stage2.lowmem.pass1.input_fp,
            state.stage2.lowmem.pass1.output_fp,
            state.stage1.output_fp,
            Fingerprints.lmdb_store(lmdb_path),
        )


# =============================================================================
# State Update Helpers
# =============================================================================


class Mark:
    """Namespace for state update operations.

    All methods are static and modify the PipelineState in place.
    Call save_state() after making changes to persist them.
    """

    @staticmethod
    def stage1_started(state: PipelineState, datasets_dir: str) -> None:
        """Record that Stage 1 is starting.

        Invalidates all downstream stages (2 and 3).

        Args:
            state: Pipeline state to update
            datasets_dir: Path to raw datasets directory
        """
        state.stage1 = StageState(input_fp=Fingerprints.raw_datasets(datasets_dir))
        state.stage2 = Stage2State()
        state.stage3 = StageState()

    @staticmethod
    def stage1_complete(state: PipelineState, stage1_dir: str) -> None:
        """Record that Stage 1 completed successfully.

        Args:
            state: Pipeline state to update
            stage1_dir: Path to stage1_mentions directory
        """
        state.stage1.output_fp = Fingerprints.stage1_out(stage1_dir)

    @staticmethod
    def stage2_started(
        state: PipelineState, stage1_dir: str, datasets: Optional[List[str]] = None
    ) -> None:
        """Record that Stage 2 is starting.

        Preserves lowmem state if it exists (for resume after interruption).
        Invalidates Stage 3.

        Args:
            state: Pipeline state to update
            stage1_dir: Path to stage1_mentions directory
            datasets: Optional list of datasets being processed
        """
        existing_lowmem = state.stage2.lowmem  # Preserve for resume capability
        state.stage2 = Stage2State(
            input_fp=Fingerprints.stage1_out(stage1_dir, datasets),
            lowmem=existing_lowmem,
        )
        state.stage3 = StageState()

    @staticmethod
    def stage2_complete(
        state: PipelineState, stage2_dir: str, datasets: Optional[List[str]] = None
    ) -> None:
        """Record that Stage 2 completed successfully.

        Args:
            state: Pipeline state to update
            stage2_dir: Path to stage2_sentiment directory
            datasets: Optional list of datasets that were processed
        """
        state.stage2.output_fp = Fingerprints.stage2_out(stage2_dir, datasets)

    @staticmethod
    def stage3_started(state: PipelineState, stage2_dir: str) -> None:
        """Record that Stage 3 is starting.

        Args:
            state: Pipeline state to update
            stage2_dir: Path to stage2_sentiment directory
        """
        state.stage3 = StageState(input_fp=Fingerprints.stage2_out(stage2_dir))

    @staticmethod
    def stage3_complete(state: PipelineState, stage3_dir: str) -> None:
        """Record that Stage 3 completed successfully.

        Args:
            state: Pipeline state to update
            stage3_dir: Path to stage3_aggregated directory
        """
        state.stage3.output_fp = Fingerprints.stage3_out(stage3_dir)

    @staticmethod
    def lowmem_pass1_started(
        state: PipelineState, stage1_dir: str, datasets: Optional[List[str]] = None
    ) -> None:
        """Record that Pass 1 (scoring to LMDB) is starting.

        Args:
            state: Pipeline state to update
            stage1_dir: Path to stage1_mentions directory
            datasets: Optional list of datasets being processed
        """
        if not state.stage2.lowmem:
            state.stage2.lowmem = LowMemState()
        state.stage2.lowmem.pass1 = StageState(
            input_fp=Fingerprints.stage1_out(stage1_dir, datasets)
        )

    @staticmethod
    def lowmem_pass1_complete(state: PipelineState, lmdb_path: str) -> None:
        """Record that Pass 1 (scoring to LMDB) completed successfully.

        Args:
            state: Pipeline state to update
            lmdb_path: Path to scores.lmdb directory
        """
        if not state.stage2.lowmem:
            state.stage2.lowmem = LowMemState()
        state.stage2.lowmem.pass1.output_fp = Fingerprints.lmdb_store(lmdb_path)
