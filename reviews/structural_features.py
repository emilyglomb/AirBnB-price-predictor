"""
Structural / metadata features from the reviews.

These describe *how many* and *how recent* a listing's reviews are, with no
NLP at all. They are instant to compute, language-independent, and usually
predictive of how active / in-demand a listing is.

Everything is aggregated to ONE row per listing so it can be joined onto the
listings table. The key column is renamed to `id` to match listings_features.csv.
"""

import pandas as pd


def build_structural_features(reviews: pd.DataFrame,
                              snapshot_date,
                              listing_id_col: str = "listing_id",
                              date_col: str = "date",
                              comment_col: str = "comment_clean") -> pd.DataFrame:
    """
    Parameters
    ----------
    reviews : cleaned reviews dataframe (output of clean_reviews).
    snapshot_date : the Inside Airbnb scrape date (the `last_scraped` value you
        found earlier). Used as the reference "today" for recency features.
        Pass a string like "2025-09-29" or a datetime.

    Returns
    -------
    DataFrame, one row per listing, key column `id`.
    """
    df = reviews.copy()
    snapshot = pd.to_datetime(snapshot_date)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_len_chars"] = df[comment_col].str.len()
    df["_len_words"] = df[comment_col].str.split().str.len()

    g = df.groupby(listing_id_col)
    feats = pd.DataFrame({
        "rev_n_reviews": g.size(),
        "rev_len_chars_mean": g["_len_chars"].mean(),
        "rev_len_words_mean": g["_len_words"].mean(),
        "_first_date": g[date_col].min(),
        "_last_date": g[date_col].max(),
    })

    # recency / activity, in days relative to the snapshot
    feats["rev_days_since_last"] = (snapshot - feats["_last_date"]).dt.days.clip(lower=0)
    feats["rev_days_since_first"] = (snapshot - feats["_first_date"]).dt.days.clip(lower=0)

    # reviews per active month (a popularity / turnover proxy)
    active_days = (feats["_last_date"] - feats["_first_date"]).dt.days
    active_months = (active_days / 30.44).clip(lower=1.0)
    feats["rev_per_month"] = feats["rev_n_reviews"] / active_months

    # reviews in the 365 days before the snapshot (recent activity)
    cutoff = snapshot - pd.Timedelta(days=365)
    recent = df[df[date_col] >= cutoff].groupby(listing_id_col).size()
    feats["rev_n_last_year"] = recent.reindex(feats.index).fillna(0).astype(int)

    feats = feats.drop(columns=["_first_date", "_last_date"])
    feats.index.name = "id"          # match listings table key
    return feats.reset_index()
