"""
Multilingual sentence-embedding features from review text.

Each review is turned into a dense vector with a multilingual
sentence-transformer, then mean-pooled per listing. These per-listing vectors
can be fed into a tree / regression model alongside the tabular features --

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 384-dimensional, multilingual, relatively fast.

"""

import os
import pandas as pd

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def compute_review_embeddings(reviews: pd.DataFrame,
                              comment_col: str = "comment_clean",
                              listing_id_col: str = "listing_id",
                              batch_size: int = 128,
                              sample_n: int | None = None,
                              cache_path: str | None = None) -> pd.DataFrame:
    """Compute one embedding vector per review.

    Returns a dataframe with [listing_id_col, emb_0 .. emb_{d-1}].
    Cached to parquet if cache_path is given.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  loading cached embeddings from {cache_path}")
        return pd.read_parquet(cache_path)

    df = reviews.copy()
    if sample_n is not None:
        df = df.head(sample_n).copy()

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  encoding {len(df):,} reviews on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    emb = model.encode(
        df[comment_col].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = emb.shape[1]
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(dim)])
    emb_df[listing_id_col] = df[listing_id_col].values

    if cache_path:
        emb_df.to_parquet(cache_path, index=False)
        print(f"  cached embeddings to {cache_path}")
    return emb_df


def build_embedding_features(review_embeddings: pd.DataFrame,
                             listing_id_col: str = "listing_id") -> pd.DataFrame:
    """Mean-pool review embeddings to one vector per listing (key `id`)."""
    feats = review_embeddings.groupby(listing_id_col).mean()
    feats.index.name = "id"
    return feats.reset_index()

