"""
Cleaning utilities for the Inside Airbnb `reviews.csv` file.

The raw reviews file has one row per review with columns:
    listing_id, id, date, reviewer_id, reviewer_name, comments

This module turns the raw, messy `comments` text into a cleaned column
(`comment_clean`) that the feature modules (structural / sentiment /
embeddings) can use.
"""

import re
import pandas as pd

# Airbnb posts an automated message when a reservation is cancelled. These are
# NOT real guest reviews (they all contain this phrase), so we drop them.
AUTOMATED_MARKER = "automated posting"

_TAG_RE = re.compile(r"<[^>]+>")   # any HTML tag, e.g. <br/> which is common here
_WS_RE = re.compile(r"\s+")        # runs of whitespace / newlines


def _strip_html(text: str) -> str:
    return _TAG_RE.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_comment(text) -> str:
    """Clean a single comment. Returns '' for missing/empty input.

    Note: we deliberately do NOT lowercase here — the sentiment and embedding
    models handle casing themselves, and casing can carry meaning. The keyword
    helper lowercases internally instead.
    """
    if not isinstance(text, str):
        return ""
    text = _strip_html(text)
    text = _normalize_whitespace(text)
    return text


def clean_reviews(reviews: pd.DataFrame,
                  comment_col: str = "comments") -> pd.DataFrame:
    """Clean the raw reviews dataframe.

    Steps:
      1. Parse the `date` column to datetime (used by structural features).
      2. Strip HTML tags and normalize whitespace in the comments.
      3. Drop rows whose comment is empty after cleaning.
      4. Remove Airbnb automated cancellation postings.

    Returns a copy with an added `comment_clean` column, index reset.
    """
    df = reviews.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["comment_clean"] = df[comment_col].apply(clean_comment)

    # drop empties
    df = df[df["comment_clean"].str.len() > 0]

    # drop automated cancellation postings
    is_automated = df["comment_clean"].str.lower().str.contains(
        AUTOMATED_MARKER, na=False
    )
    df = df[~is_automated]

    return df.reset_index(drop=True)
