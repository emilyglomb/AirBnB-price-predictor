"""
Build the per-listing review feature table for the Florence Airbnb project.

Run from the repo root:

    python -m reviews.build_reviews_features

Reads   data/reviews.csv
Writes  data/reviews_features.csv   (one row per listing, key column `id`)

which is then joined onto listings_features.csv on `id` (see README at bottom).

Toggle the feature sets, the snapshot date, and the test sample in CONFIG below.
"""

import os
import pandas as pd

from reviews.clean_reviews import clean_reviews
from reviews.structural_features import build_structural_features
from reviews.sentiment_features import (
    compute_review_sentiment, build_sentiment_features,
)
from reviews.embedding_features import (
    compute_review_embeddings, build_embedding_features,
)

# ------------------------------- CONFIG ----------------------------------
REVIEWS_PATH = "data/reviews.csv"
OUTPUT_PATH = "data/reviews_features.csv"

SNAPSHOT_DATE = "2025-09-24"       

USE_STRUCTURAL = True
USE_SENTIMENT = True
USE_EMBEDDINGS = False              # advanced; we will turn on once the rest works

# we set to an int (e.g. 2000) to test the slow steps on a small sample first,
# then set back to None for the real run.
SAMPLE_N = None

SENT_CACHE = "data/_cache_review_sentiment.parquet"
EMB_CACHE = "data/_cache_review_embeddings.parquet"
# -------------------------------------------------------------------------


def main():
    print("Loading reviews ...")
    reviews = pd.read_csv(REVIEWS_PATH)
    print(f"  {len(reviews):,} raw reviews")

    print("Cleaning ...")
    reviews = clean_reviews(reviews)
    print(f"  {len(reviews):,} reviews after cleaning")

    feature_tables = []

    if USE_STRUCTURAL:
        print("Structural features ...")
        feature_tables.append(
            build_structural_features(reviews, snapshot_date=SNAPSHOT_DATE)
        )

    if USE_SENTIMENT:
        print("Sentiment features (slow) ...")
        with_sent = compute_review_sentiment(
            reviews, sample_n=SAMPLE_N, cache_path=SENT_CACHE
        )
        feature_tables.append(build_sentiment_features(with_sent))

    if USE_EMBEDDINGS:
        print("Embedding features (slow) ...")
        emb = compute_review_embeddings(
            reviews, sample_n=SAMPLE_N, cache_path=EMB_CACHE
        )
        feature_tables.append(build_embedding_features(emb))

    if not feature_tables:
        raise SystemExit("No feature set enabled — turn one on in CONFIG.")

    print("Merging feature tables on `id` ...")
    features = feature_tables[0]
    for tbl in feature_tables[1:]:
        features = features.merge(tbl, on="id", how="outer")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Done -> {OUTPUT_PATH}: {features.shape[0]:,} listings, "
          f"{features.shape[1]} columns")


if __name__ == "__main__":
    main()
