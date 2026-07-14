# Airbnb Price Prediction — Florence

Final project for the SoSe 2026 Machine Learning course.
We predict nightly listing prices for Airbnb accommodations in Florence, Italy,
combining multiple data modalities: tabular listing data, review text, and spatial coordinates.

---

## Data

The data is **not** included in this repository (files are too large for git).
It comes from the open-data project [Inside Airbnb](https://insideairbnb.com/get-the-data).

Download the **Florence, Italy** snapshot dated **2025-09-24** — use this exact date,
since Inside Airbnb re-scrapes several times a year and different snapshots will not match our outputs.

Download and unzip the following files into a `data/` folder at the repo root:

```
data/
├── listings.csv      # detailed listings  (from listings.csv.gz)
├── reviews.csv       # detailed reviews   (from reviews.csv.gz)
└── calendar.csv      # calendar           (from calendar.csv.gz)
```

The `data/` folder is gitignored and will not be committed.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## How to reproduce results

Run the steps **in order**:

**Step 1 — Tabular feature engineering**

Open and run `cleaning.ipynb` from top to bottom.
Output: `data/listings_features.csv`

**Step 2 — Review text features**

```bash
python -m reviews.build_reviews_features
```

This takes ~2 hours on a GPU (Google Colab T4 recommended for the sentiment step).
Output: `data/reviews_features.csv`

**Step 3 — Modelling**

Open and run `model.ipynb` from top to bottom.
It reads both feature CSVs, merges them, trains and evaluates all models,
and produces SHAP importance plots.

---

## Repository structure

```
cleaning.ipynb                  # tabular cleaning & feature engineering
model.ipynb                     # all models, evaluation, SHAP analysis
reviews/
    build_reviews_features.py   # entry point — run this for Step 2
    clean_reviews.py            # strip HTML, drop empty, remove automated postings
    structural_features.py      # counts & recency features (no NLP)
    sentiment_features.py       # multilingual BERT sentiment (1–5 stars per review)
    embedding_features.py       # multilingual sentence embeddings (optional)
requirements.txt
```

---
