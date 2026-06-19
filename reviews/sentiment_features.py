"""
Multilingual sentiment features from review text.

Florence reviews are written in many languages (Italian, English, French,
German, Spanish, ...), so we use a *multilingual* model rather than an
English-only keyword list or lexicon.

Model: nlptown/bert-base-multilingual-uncased-sentiment
  - predicts a star rating from 1 to 5
  - trained on reviews in en / nl / de / fr / es / it  -> good fit for Florence

For each review we compute an EXPECTED star value (a continuous 1-5 score =
sum of star * probability), then aggregate per listing.

WARNING -- this runs a transformer over every review. On CPU with several
hundred thousand reviews it takes hours. Recommendations:
  * run it on a GPU (e.g. Google Colab),
  * test with sample_n=2000 first,
  * rely on the parquet cache so you only pay the cost once.
"""

import os
import pandas as pd

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


def _expected_stars(score_dicts) -> float:
    """Turn a list of {label, score} for the 5 star-classes into an expected
    star value. Labels look like '1 star', '2 stars', ..."""
    exp = 0.0
    for d in score_dicts:
        stars = int(d["label"].split()[0])
        exp += stars * d["score"]
    return exp


def compute_review_sentiment(reviews: pd.DataFrame,
                             comment_col: str = "comment_clean",
                             batch_size: int = 64,
                             sample_n: int | None = None,
                             cache_path: str | None = None) -> pd.DataFrame:
    """Compute a per-review sentiment score (expected stars, 1-5).

    Returns the dataframe with an added `sent_stars` column.
    If `cache_path` exists it is loaded instead of recomputing.
    If `sample_n` is set, only that many reviews are processed (for testing).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  loading cached sentiment from {cache_path}")
        return pd.read_parquet(cache_path)

    df = reviews.copy()
    if sample_n is not None:
        df = df.head(sample_n).copy()

    # heavy imports kept local so the rest of the pipeline runs without
    # transformers / torch installed
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"  running sentiment on {'GPU' if device == 0 else 'CPU'} "
          f"over {len(df):,} reviews ...")

    clf = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        top_k=None,            # return all 5 class scores
        truncation=True,
        max_length=512,
        device=device,
        batch_size=batch_size,
    )

    results = clf(df[comment_col].tolist())     # one list-of-dicts per review
    df["sent_stars"] = [_expected_stars(r) for r in results]

    if cache_path:
        df.to_parquet(cache_path, index=False)
        print(f"  cached sentiment to {cache_path}")
    return df


def build_sentiment_features(reviews_with_sent: pd.DataFrame,
                             listing_id_col: str = "listing_id",
                             sent_col: str = "sent_stars") -> pd.DataFrame:
    """Aggregate per-review sentiment to one row per listing (key `id`)."""
    df = reviews_with_sent.copy()
    df["_neg"] = (df[sent_col] < 3).astype(int)

    g = df.groupby(listing_id_col)
    feats = pd.DataFrame({
        "rev_sent_mean": g[sent_col].mean(),
        "rev_sent_std": g[sent_col].std(),
        "rev_sent_min": g[sent_col].min(),
        "rev_frac_negative": g["_neg"].mean(),
    })
    feats.index.name = "id"
    return feats.reset_index()
