# Airbnb Price Prediction — Florence

Final project for the SoSe 2026 Machine Learning course. We predict nightly
listing prices for Airbnb accommodations in Florence, Italy, combining multiple
data modalities (tabular listing data, review text, spatial coordinates).

## Data

The data is **not** included in this repository (the files are far too large
for git — the reviews file alone is hundreds of MB). It is downloaded from the
open-data project [Inside Airbnb](https://insideairbnb.com/get-the-data).

To reproduce our results, download the **Florence, Italy** snapshot from
**[2025-09-24]** — use this exact date, since Inside Airbnb
re-scrapes several times a year and different snapshots will not match our
cleaned outputs.

Download these files, unzip them, and place them in a `data/` folder at the
repo root:
data/
    ├── listings.csv      # detailed listings (from listings.csv.gz)
    ├── reviews.csv       # detailed reviews  (from reviews.csv.gz)
    └── calendar.csv      # calendar          (from calendar.csv.gz)
    .
    .
    .
    
The `data/` folder is gitignored and will not be committed.

## Setup

    pip install -r requirements.txt

(Python 3.x, with pandas, numpy, matplotlib, scikit-learn — see requirements.txt.)

## Repository structure

    cleaning.ipynb       # tabular cleaning & feature engineering -> data/listings_features.csv
    reviews/             # review-text processing & features (joined on listing id)
    ...                  # (modeling, evaluation — added as the project grows)

