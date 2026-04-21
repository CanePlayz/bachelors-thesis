"""Model loading and batch scoring for sentiment analysis.

This module provides GPU-accelerated model loading and scoring for
HuggingFace sentiment models. It handles:
- CUDA/CPU device selection
- OOM-resilient batch size halving
- Proper model disposal to free memory
- Support for both targeted and non-targeted models

Key functions:
- load_sentiment_model(): Load a HuggingFace classifier pipeline
- score_texts(): Score a batch of texts with a loaded model
- score_texts_targeted(): Score texts with a target ticker
- dispose_model(): Clean up model resources

Performance note:
    For maximum efficiency, load each model once, score all texts,
    then dispose before loading the next model. This minimizes
    model loading overhead.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import logging
import os
import sys
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from data.config import MODEL_MAX_LEN
from transformers import pipeline

try:
    # Silence noisy pipeline warnings like "Device set to use cuda:0" *and*
    # the "Loading weights: 100%|...|" tqdm bar emitted by transformers'
    # internal model loader (which writes raw to stderr and breaks our
    # indented output).
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass
except Exception:
    pass

try:
    # Silence the HF Hub "Loading weights: 100% |...|" tqdm bar that breaks
    # our indented output. Authentication warnings from huggingface_hub are
    # also suppressed so missing HF_TOKEN doesn't dump a banner mid-pipeline.
    from huggingface_hub.utils import logging as _hf_hub_logging
    from huggingface_hub.utils.tqdm import (
        disable_progress_bars as _hf_disable_progress_bars,
    )

    _hf_disable_progress_bars()
    _hf_hub_logging.set_verbosity_error()
except Exception:
    pass

# Suppress torch.compile symbolic shape guard warnings
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr at the file descriptor level.

    This catches C-level prints (e.g., from CUDA/PyTorch) that bypass
    Python's sys.stdout/sys.stderr. Necessary for suppressing messages like:
    - "Device set to use cuda:0"
    - "You seem to be using the pipelines sequentially on GPU..."
    """
    # Save the original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    # Open /dev/null (or NUL on Windows)
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        # Redirect stdout and stderr to devnull at the FD level
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        # Close the duplicated FDs
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull)


def get_device() -> int:
    """Get the best available device for model inference.

    Returns:
        Device ID: 0 for CUDA GPU, -1 for CPU
    """
    if torch.cuda.is_available():
        return 0  # Use first CUDA GPU
    return -1  # Fall back to CPU


def load_sentiment_model(model_name: str, device: Optional[int] = None):
    """Load a HuggingFace text-classification pipeline.

    Args:
        model_name: HuggingFace model identifier
        device: Device to use (-1=CPU, 0+=GPU). If None, auto-detect.

    Returns:
        Loaded HuggingFace pipeline ready for inference
    """
    # Suppress HuggingFace advisory warnings about GPU usage
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # If the user has set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in their
    # environment, log in silently so HF Hub uses authenticated requests
    # (higher rate limits, no warning banner). No-op when no token is set.
    _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if _hf_token:
        try:
            from huggingface_hub import login as _hf_login

            _hf_login(token=_hf_token, add_to_git_credential=False)
        except Exception:
            pass

    # Auto-detect device if not specified
    if device is None:
        device = get_device()

    # Load pipeline with truncation enabled, suppressing CUDA/GPU messages.
    # Use fp16 on GPU for ~2x memory savings and faster inference on Tensor Cores.
    model_dtype = torch.float16 if device >= 0 else torch.float32  # type: ignore[attr-defined]

    clf = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        device=device,
        dtype=model_dtype,
    )

    return clf


def dispose_model(model: Any) -> None:
    """Clean up model resources to free GPU/CPU memory.

    Attempts to delete the model and run garbage collection.
    This is a best-effort cleanup - not guaranteed to free all memory.

    Args:
        model: The model/pipeline object to dispose
    """
    try:
        # Move model to CPU first to free GPU memory
        if hasattr(model, "model"):
            model.model.to("cpu")
    except Exception:
        pass

    try:
        del model
    except Exception:
        pass

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def label_to_score(label: str, confidence: float) -> float:
    """Convert a sentiment label + confidence to a signed score.

    Maps model outputs to a consistent [-1, +1] scale:
    - Positive labels -> +confidence
    - Negative labels -> -confidence
    - Neutral labels -> 0.0

    Args:
        label: Model's predicted label
        confidence: Prediction confidence (0-1)

    Returns:
        Signed sentiment score in [-1, +1]
    """
    lab = label.lower()

    # Positive labels (including FinTwitBERT's "bullish")
    if lab in {"positive", "pos", "label_2", "label_3", "bullish", "bull"}:
        return +confidence

    # Negative labels (including FinTwitBERT's "bearish")
    if lab in {"negative", "neg", "label_0", "bearish", "bear"}:
        return -confidence

    # Neutral labels
    if lab in {"neutral", "neu", "label_1"}:
        return 0.0

    # Unknown label - return 0 to be safe
    return 0.0


def normalize_model_output(output: Any) -> Tuple[float, str]:
    """Normalize HuggingFace classifier output to (score, label).

    Handles single-output format (top_k=1, the default).
    Converts the best prediction to a signed sentiment score.

    Args:
        output: Raw model output (dict with 'label' and 'score')

    Returns:
        Tuple of (signed_score, best_label)
    """
    # Handle single dict output (top_k=1 default)
    if isinstance(output, Mapping) and "label" in output:
        label = str(output.get("label", "unknown"))
        conf = float(output.get("score", 0.0))
        return (label_to_score(label, conf), label)

    # Handle list output (shouldn't happen with top_k=1, but be safe)
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for item in output:
            if isinstance(item, Mapping) and "label" in item:
                label = str(item.get("label", "unknown"))
                conf = float(item.get("score", 0.0))
                return (label_to_score(label, conf), label)

    return (0.0, "unknown")


def _effective_max_length(clf: Any, requested_max_length: int) -> int:
    """Choose a safe max_length for this pipeline/model.

    Some tokenizers expose a smaller `model_max_length` than the global config
    (e.g., 128). Passing a larger `max_length` can emit warnings and, for some
    models, risk indexing errors.

    We therefore clamp to the minimum of:
    - requested_max_length (pipeline config upper bound)
    - tokenizer.model_max_length (if it looks reasonable)
    - model.config.max_position_embeddings (if available)
    """

    effective = max(1, int(requested_max_length))

    try:
        tok = getattr(clf, "tokenizer", None)
        tok_max = getattr(tok, "model_max_length", None)
        if tok_max is not None:
            tok_max_int = int(tok_max)
            # Some tokenizers use a huge sentinel for "unbounded".
            if 1 <= tok_max_int <= 100000:
                effective = min(effective, tok_max_int)
    except Exception:
        pass

    try:
        model = getattr(clf, "model", None)
        cfg = getattr(model, "config", None)
        mpe = getattr(cfg, "max_position_embeddings", None)
        if mpe is not None:
            mpe_int = int(mpe)
            if 1 <= mpe_int <= 100000:
                effective = min(effective, mpe_int)
    except Exception:
        pass

    return max(1, effective)


def score_batch_internal(
    clf,
    texts: List[str],
    batch_size: int,
    max_length: int = MODEL_MAX_LEN,
) -> List[Tuple[float, str]]:
    """Internal batch scoring with OOM retry logic.

    If a CUDA OOM error occurs, retries with halved batch size.

    Args:
        clf: Loaded HuggingFace pipeline
        texts: List of texts to score
        batch_size: Initial batch size
        max_length: Maximum token length for truncation

    Returns:
        List of (score, label) tuples, one per input text
    """
    if not texts:
        return []

    bs = max(1, batch_size)
    eff_max_length = _effective_max_length(clf, max_length)

    while True:
        try:
            # Run inference (top_k=1 default - only best prediction, faster)
            raw_outputs = clf(
                texts,
                batch_size=bs,
                truncation=True,
                max_length=eff_max_length,
            )

            # Normalize each output
            results = [normalize_model_output(out) for out in raw_outputs]
            return results

        except RuntimeError as exc:
            msg = str(exc).lower()

            # If OOM and we can reduce batch size, retry
            if "out of memory" in msg and bs > 1:
                bs = max(1, bs // 2)
                # Clear CUDA cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # Non-OOM error or can't reduce further - re-raise
            raise


def score_batch_targeted_internal(
    clf,
    items: List[Tuple[str, str]],
    batch_size: int,
    max_length: int = MODEL_MAX_LEN,
) -> List[Tuple[float, str]]:
    """Internal batch scoring for targeted models.

    For models like twitter-roberta-base-topic-sentiment that accept
    a text_pair parameter for topic-conditioned sentiment.

    Args:
        clf: Loaded HuggingFace pipeline
        items: List of (text, target) tuples
        batch_size: Initial batch size
        max_length: Maximum token length

    Returns:
        List of (score, label) tuples
    """
    if not items:
        return []

    # Format inputs for targeted model (text + text_pair)
    # The Cardiff topic-sentiment model expects {"text": ..., "text_pair": ...}
    formatted_inputs = [{"text": text, "text_pair": target} for text, target in items]

    bs = max(1, batch_size)
    eff_max_length = _effective_max_length(clf, max_length)

    while True:
        try:
            # Run inference (top_k=1 default - only best prediction, faster)
            raw_outputs = clf(
                formatted_inputs,
                batch_size=bs,
                truncation=True,
                max_length=eff_max_length,
            )

            results = [normalize_model_output(out) for out in raw_outputs]
            return results

        except RuntimeError as exc:
            msg = str(exc).lower()

            if "out of memory" in msg and bs > 1:
                bs = max(1, bs // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            raise


class SentimentScorer:
    """Wrapper class for scoring texts with a specific model.

    Manages model lifecycle and provides convenient scoring methods.
    Use as a context manager for automatic cleanup:

        with SentimentScorer("model-name") as scorer:
            results = scorer.score_texts(["text1", "text2"])
    """

    def __init__(self, model_name: str, batch_size: int = 32):
        """Initialize scorer with a model.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Default batch size for scoring
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.device = get_device()

    def _ensure_loaded(self) -> None:
        """Lazy-load the model if not already loaded."""
        if self.model is None:
            self.model = load_sentiment_model(self.model_name, self.device)

    def score_texts(self, texts: List[str]) -> List[Tuple[float, str]]:
        """Score a list of texts.

        Args:
            texts: List of texts to score

        Returns:
            List of (score, label) tuples
        """
        self._ensure_loaded()
        return score_batch_internal(self.model, texts, self.batch_size)

    def score_texts_targeted(
        self, items: List[Tuple[str, str]]
    ) -> List[Tuple[float, str]]:
        """Score texts with target tickers.

        Args:
            items: List of (text, target) tuples

        Returns:
            List of (score, label) tuples
        """
        self._ensure_loaded()
        return score_batch_targeted_internal(self.model, items, self.batch_size)

    def dispose(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            dispose_model(self.model)
            self.model = None

    def __enter__(self) -> "SentimentScorer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disposes model."""
        self.dispose()


def iter_batches(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Iterate over items in batches.

    Args:
        items: List to batch
        batch_size: Size of each batch

    Yields:
        Lists of items, each up to batch_size elements
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
