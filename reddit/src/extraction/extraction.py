"""Fast ticker extraction logic for v2 pipeline (src).

This module implements multi-phase ticker extraction from text:
1. $-prefixed tickers (e.g., $AAPL) - highest confidence
2. Bare ALL-CAPS tokens (e.g., AAPL) - requires stoplist filtering
3. Company names via Aho-Corasick automaton - O(n) matching
4. WSB-specific aliases (slang terms like "GAMESTONK")

The extraction is designed to minimize false positives while
capturing the various ways Reddit users reference stocks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from data.ticker_universe import STOPLIST_TOKENS, TickerUniverse

# Pattern to match URLs (for stripping before ticker extraction)
# Matches http://, https://, and www. URLs
URL_PATTERN = re.compile(
    r"https?://[^\s\)\]>]+|www\.[^\s\)\]>]+",
    re.IGNORECASE,
)

# Pattern for $-prefixed tickers (highest confidence)
# Matches: $AAPL, $GME, $TSLA (1-5 letters after $)
# Requires non-alphanumeric boundaries
DOLLAR_TICKER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,5})(?![A-Za-z0-9])")

# Pattern for bare ALL-CAPS tokens (3-5 letters)
# Matches: AAPL, TSLA (3-5 letter symbols). Two-letter tokens are treated
# as higher-risk and require the `$` prefix (same behaviour as 1-letter tickers).
BARE_CAPS_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Z]{3,5})(?![A-Za-z0-9])")

# WSB-specific aliases (slang terms for stocks)
# Maps ticker symbols to list of slang/nickname strings
# Note: Company names come from clean_tickers.csv
#
# IMPORTANT: Only include aliases that unambiguously refer to the STOCK, not:
# - CEO names (Elon could be SpaceX/Twitter/xAI context, not TSLA)
# - Generic memes that might not indicate stock discussion
# Safe aliases: stock-specific slang, old tickers, and common company names
# that differ from the official ticker/parent company name.
WSB_ALIASES: Dict[str, List[str]] = {
    # Meme stock slang
    "GME": ["GAMESTONK", "GAME STONK"],
    # Rebranded companies (old name/ticker still widely used)
    "META": ["FACEBOOK", "FB"],
    "GOOGL": ["GOOGLE", "ALPHABET"],  # Everyone says "Google stock"
    "XYZ": ["SQUARE", "SQ"],  # Block was formerly Square, old ticker SQ
    # Company names that differ significantly from ticker
    "BABA": ["ALIBABA", "ALI BABA"],
    "KO": ["COCA-COLA", "COCA COLA", "COKE"],  # Ticker is KO, not obvious
    "BRK.A": ["BERKSHIRE"],
    "BRK.B": ["BERKSHIRE"],
    # Product names more famous than company name (Snap Inc → Snapchat)
    "SNAP": ["SNAPCHAT"],
    # Abbreviations commonly used
    "JNJ": ["J&J"],
}

# Pre-compiled alias patterns for efficient matching
# Built once at module load time
_WSB_ALIAS_PATTERNS: Dict[str, List[Tuple[str, re.Pattern]]] = {}


def _build_wsb_alias_patterns() -> None:
    """Pre-compile all WSB alias patterns once at module load.

    Creates case-insensitive regex patterns with word boundaries
    for each alias string.
    """
    global _WSB_ALIAS_PATTERNS

    # Iterate over each ticker and its aliases
    for ticker, aliases in WSB_ALIASES.items():
        patterns = []

        # Compile a pattern for each alias
        for alias in aliases:
            if alias:
                # Create word-boundary pattern, case-insensitive
                pat = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                patterns.append((alias, pat))

        _WSB_ALIAS_PATTERNS[ticker] = patterns


# Build patterns at module load time
_build_wsb_alias_patterns()


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if match at [start:end] has proper word boundaries.

    Used to validate Aho-Corasick matches to ensure they're
    complete words rather than substrings.

    Args:
        text: The full text being searched
        start: Start index of the match
        end: End index of the match (exclusive)

    Returns:
        True if the match has word boundaries on both sides
    """
    # Check left boundary
    if start > 0:
        left_char = text[start - 1]
        # If left char is alphanumeric or underscore, not a boundary
        if left_char.isalnum() or left_char == "_":
            return False

    # Check right boundary
    if end < len(text):
        right_char = text[end]
        # If right char is alphanumeric or underscore, not a boundary
        if right_char.isalnum() or right_char == "_":
            return False

    return True


def _strip_urls(text: str) -> str:
    """Remove URLs from text to prevent false positive ticker matches.

    URLs often contain tracking parameters (e.g., ftag=CNM-00) that
    could match ticker symbols. Stripping them avoids these false positives.
    """
    return URL_PATTERN.sub(" ", text)


def extract_tickers_fast(
    text: str,
    universe: TickerUniverse,
    tickers_only: bool = False,
) -> List[Tuple[str, str, str]]:
    """Fast ticker extraction with strict rules to avoid false positives.

    Uses Aho-Corasick algorithm for O(text_length + matches) company name matching.

    Args:
        text: The text to extract tickers from
        universe: The TickerUniverse with lookup structures
        tickers_only: If True, skip company name (Phase 3) and alias (Phase 4)
                      matching. Only match $-prefixed and bare ALL-CAPS tickers.

    Returns:
        List of (ticker, match_type, matched_on) tuples where:
        - ticker: the stock symbol (e.g., 'AAPL')
        - match_type: "ticker_dollar", "ticker", "name", or "alias"
        - matched_on: the exact string that triggered the match
    """
    # Return empty list for empty input
    if not text:
        return []

    # Strip URLs to prevent false positives from URL parameters
    text = _strip_urls(text)

    # Dict to store matches (deduplicates by ticker)
    # Value is (match_type, matched_on) tuple
    matches: Dict[str, Tuple[str, str]] = {}

    # --- Phase 1: $-prefixed tickers (highest confidence) ---
    # These are explicit stock references like $AAPL
    # NOTE: We match cashtags regardless of STOPLIST_TOKENS because
    # $SYMBOL is an intentional stock reference. The stoplist only
    # filters bare ALL-CAPS tokens which might be regular words.
    for m in DOLLAR_TICKER_PATTERN.finditer(text):
        # Extract symbol (uppercase)
        sym = m.group(1).upper()

        # Only accept if it's a known ticker
        if sym in universe.all_symbols:
            # Get the full matched string (including $)
            matched_str = m.group(0)
            matches[sym] = ("ticker_dollar", matched_str)

    # --- Phase 2: Bare ALL-CAPS tokens (3-5 letters) ---
    # Anything shorter than 3 letters is too risky without $ prefix
    # These are uppercase tokens like AAPL without $
    for m in BARE_CAPS_PATTERN.finditer(text):
        sym = m.group(1)

        # Skip if already matched
        if sym in matches:
            continue

        # Skip common words/abbreviations (stoplist)
        if sym in STOPLIST_TOKENS:
            continue

        # Only accept known tickers
        if sym not in universe.all_symbols:
            continue

        # Add to matches
        matches[sym] = ("ticker", sym)

    # Skip phases 3 & 4 if tickers_only mode
    if tickers_only:
        return [(t, mt, mo) for t, (mt, mo) in matches.items()]

    # --- Phase 3: Company names via Aho-Corasick ---
    # Searches for full company names like "Apple Inc".
    # Problematic tickers (common words as company names) are already
    # excluded from the automaton at build time in ticker_universe.py.
    # Matches are not case-sensitive.
    text_upper = text.upper()

    # Iterate over all matches from the automaton
    for end_idx, (
        normalized_name,
        ticker,
        _,  # original_name (not needed here)
    ) in universe.name_automaton.iter(text_upper):

        # Skip if already matched by symbol
        if ticker in matches:
            continue

        # Skip short tickers (1-2 letters) from name matches
        # These require explicit $-prefix for safety
        if len(ticker) <= 2:
            continue

        # Calculate start index from end index
        start_idx = end_idx - len(normalized_name) + 1

        # Verify word boundaries to avoid partial matches
        if not _is_word_boundary(text_upper, start_idx, end_idx + 1):
            continue

        # Get the matched text from original (preserves case)
        matched_text = text[start_idx : end_idx + 1]
        matches[ticker] = ("name", matched_text)

    # --- Phase 4: WSB-specific aliases ---
    # Searches for slang terms like "GAMESTONK" for GME
    for ticker, alias_patterns in _WSB_ALIAS_PATTERNS.items():

        # Skip if already matched
        if ticker in matches:
            continue

        # Skip if ticker not in universe and not in aliases
        if ticker not in universe.all_symbols and ticker not in WSB_ALIASES:
            continue

        # Skip if on stoplist
        if ticker in STOPLIST_TOKENS:
            continue

        # Try each alias pattern
        for alias, pattern in alias_patterns:
            # Quick check: is the alias text present?
            if alias.upper() not in text_upper:
                continue

            # Full regex match
            m = pattern.search(text)
            if m:
                matches[ticker] = ("alias", m.group(0))
                break  # Only need one alias match

    # Convert dict to list of tuples
    return [(t, mt, mo) for t, (mt, mo) in matches.items()]
