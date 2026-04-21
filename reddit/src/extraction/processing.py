"""Record types and processing functions for v2 pipeline (src).

This module defines the MentionRecord data class and provides
functions to process Reddit submissions and comments, extracting
ticker mentions and creating structured records.

Processing flow:
1. Extract raw text from submission/comment
2. Filter deleted/removed content
3. Filter moderator/bot posts (would inflate counts with boilerplate text)
4. Apply ticker extraction
5. Build MentionRecord with metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from data.ticker_universe import TickerUniverse

from .extraction import extract_tickers_fast

# Bot/moderator accounts to filter out
# These post automated messages containing ticker symbols that are not relevant to our analysis
BOT_AUTHORS = {
    "automoderator",
    "auto-moderator",
    "mod_team",
    "qualityvote",
    "superstonk_qv",  # Superstonk quality vote bot
}

# Author suffixes that indicate bot accounts
BOT_AUTHOR_SUFFIXES = ("bot", "_bot", "-bot", "_qv")


def is_bot_or_mod(row: dict) -> bool:
    """Check if a post is from a bot or moderator.

    Filters out:
    - Distinguished moderator posts
    - Stickied posts (usually automated)
    - Known bot accounts (AutoModerator, etc.)
    - Authors with bot-like name patterns

    Args:
        row: Raw submission/comment data

    Returns:
        True if post should be filtered out
    """
    # Check if distinguished as moderator
    if row.get("distinguished") == "moderator":
        return True

    # Check if stickied (usually mod/bot posts)
    if row.get("stickied") is True:
        return True

    # Check author against known bots
    author = (row.get("author") or "").lower()
    if author in BOT_AUTHORS:
        return True

    # Check author name patterns
    if author.endswith(BOT_AUTHOR_SUFFIXES):
        return True

    return False


@dataclass
class MentionRecord:
    """A single Reddit post/comment with ticker mentions.

    This is the core data structure output by the pipeline.
    Each record represents one submission or comment that
    contains at least one ticker mention.

    Attributes:
        id: Reddit fullname (e.g., 't3_abc123' for submission)
        subreddit: Name of the subreddit
        author: Reddit username of the poster
        created_utc: Unix timestamp of creation
        type: Either 'submission' or 'comment'
        text: The text content (truncated to 10k chars)
        title: Submission title (None for comments)
        score: Reddit score (upvotes - downvotes)
        num_comments: Number of comments (None for comments)
        tickers: List of matched ticker symbols
        match_types: Dict mapping ticker to how it was matched
        matched_on: Dict mapping ticker to the matched string
    """

    id: str
    subreddit: str
    author: str
    created_utc: int
    type: str  # "submission" or "comment"

    text: str
    title: Optional[str]

    score: int
    num_comments: Optional[int]

    tickers: List[str]
    match_types: Dict[str, str]
    matched_on: Dict[str, str]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict.

        Returns:
            Dictionary with all record fields
        """
        return {
            "id": self.id,
            "subreddit": self.subreddit,
            "author": self.author,
            "created_utc": self.created_utc,
            "type": self.type,
            "text": self.text,
            "title": self.title,
            "score": self.score,
            "num_comments": self.num_comments,
            "tickers": self.tickers,
            "match_types": self.match_types,
            "matched_on": self.matched_on,
        }


def process_submission(
    row: dict,
    universe: TickerUniverse,
    min_text_len: int = 0,
    tickers_only: bool = False,
) -> Optional[MentionRecord]:
    """Process a submission and extract ticker mentions.

    Combines title and selftext for matching. Requires context
    for ambiguous tickers (uses require_context_for_ambiguous=True).

    Args:
        row: Raw submission data from NDJSON
        universe: TickerUniverse for matching
        min_text_len: Minimum combined text length to process
        tickers_only: If True, only match $-prefixed and bare caps tickers

    Returns:
        MentionRecord if tickers found, None otherwise
    """
    # Skip bot/moderator posts (would inflate counts with boilerplate text)
    if is_bot_or_mod(row):
        return None

    # Extract title and selftext
    title = row.get("title", "") or ""
    selftext = row.get("selftext", "") or ""

    # Clear deleted/removed content
    if selftext in ("[deleted]", "[removed]"):
        selftext = ""

    # Combine title and selftext for matching
    combined_text = f"{title} {selftext}".strip()

    # Skip if too short
    if len(combined_text) < min_text_len:
        return None

    # Extract tickers (require context for submissions)
    matches = extract_tickers_fast(
        combined_text,
        universe,
        tickers_only=tickers_only,
    )

    # Skip if no matches
    if not matches:
        return None

    # Parse timestamp (handle string or int)
    created_utc = row.get("created_utc")
    if isinstance(created_utc, str):
        created_utc = int(float(created_utc))
    elif created_utc is None:
        created_utc = 0

    # Build and return record
    return MentionRecord(
        id=row.get("name", f"t3_{row.get('id', 'unknown')}"),
        subreddit=row.get("subreddit", "unknown"),
        author=row.get("author", "[deleted]"),
        created_utc=created_utc,
        type="submission",
        text=combined_text[:10000],  # Truncate to 10k chars
        title=title,
        score=row.get("score", 0) or 0,
        num_comments=row.get("num_comments", 0) or 0,
        tickers=[t for t, _, _ in matches],
        match_types={t: mt for t, mt, _ in matches},
        matched_on={t: mo for t, _, mo in matches},
    )


def process_comment(
    row: dict,
    universe: TickerUniverse,
    min_text_len: int = 0,
    tickers_only: bool = False,
) -> Optional[MentionRecord]:
    """Process a comment and extract ticker mentions.

    Uses comment body for matching. Does not require context
    for ambiguous tickers (uses require_context_for_ambiguous=False).

    Args:
        row: Raw comment data from NDJSON
        universe: TickerUniverse for matching
        min_text_len: Minimum body length to process
        tickers_only: If True, only match $-prefixed and bare caps tickers

    Returns:
        MentionRecord if tickers found, None otherwise
    """
    # Skip bot/moderator posts (would inflate counts with boilerplate text)
    if is_bot_or_mod(row):
        return None

    # Extract body text
    body = row.get("body", "") or ""

    # Skip deleted/removed comments
    if body in ("[deleted]", "[removed]"):
        return None

    # Skip if too short
    if len(body) < min_text_len:
        return None

    # Extract tickers (no context required for comments)
    matches = extract_tickers_fast(body, universe, tickers_only=tickers_only)

    # Skip if no matches
    if not matches:
        return None

    # Parse timestamp (handle string or int)
    created_utc = row.get("created_utc")
    if isinstance(created_utc, str):
        created_utc = int(float(created_utc))
    elif created_utc is None:
        created_utc = 0

    # Build and return record
    return MentionRecord(
        id=row.get("name", f"t1_{row.get('id', 'unknown')}"),
        subreddit=row.get("subreddit", "unknown"),
        author=row.get("author", "[deleted]"),
        created_utc=created_utc,
        type="comment",
        text=body[:10000],  # Truncate to 10k chars
        title=None,  # Comments don't have titles
        score=row.get("score", 0) or 0,
        num_comments=None,  # Not applicable for comments
        tickers=[t for t, _, _ in matches],
        match_types={t: mt for t, mt, _ in matches},
        matched_on={t: mo for t, _, mo in matches},
    )
