# Pune Real Estate Market Price Analysis and Prediction

An end-to-end B.Tech AIML major project for predicting Pune residential property prices across two targets:

- Monthly rental price in INR/month
- Purchase price in INR

The project includes data collection scaffolding, a realistic synthetic-data fallback, data cleaning, EDA, feature engineering, dual-model training, SHAP explainability, and a Streamlit dashboard.

## Project Structure

```text
pune-realestate/
├── data/
│   ├── raw/
│   └── cleaned/
├── models/
├── notebooks/
│   ├── EDA.ipynb
│   └── figures/
├── src/
│   ├── cleaning/
│   ├── dashboard/
│   ├── eda/
│   ├── features/
│   ├── models/
│   └── scraper/
├── .streamlit/
├── requirements.txt
└── README.md
```

## Architecture

```text
Raw data sources
   |--  scraper
        |
        v
data/raw/*.csv
        |
        v
Cleaning pipeline
   |-- merge
   |-- fuzzy deduplicate
   |-- impute
   |-- outlier handling
   |-- encode + normalize
        |
        v
data/cleaned/pune_cleaned*.csv
        |
        v
Feature engineering
   |-- price_per_sqft
   |-- floor_ratio
   |-- connectivity_score
   |-- locality_tier
   |-- area_per_bhk
        |
        v
data/cleaned/pune_feature_engineered*.csv
        |
        +--> EDA plots
        +--> Rental model training
        +--> Purchase model training
                    |
                    v
             models/*.pkl + metrics.json
                    |
                    v
           Streamlit dashboard + SHAP explanations
```

## Features Covered

The pipeline captures the requested property attributes across physical, furnishing, society, location, and market dimensions, including BHK, area, floors, building age, amenities, furnishing, locality, metro and IT-park distance, rent, purchase price, price per sqft, brokerage type, and listing freshness.

## Setup

1. Open a terminal in the project root:
   `D:\VS code\pune-realestate`
2. Create a virtual environment:
   `python -m venv .venv`
3. Activate it on Windows PowerShell:
   `.venv\Scripts\Activate.ps1`
4. Install dependencies:
   `pip install -r requirements.txt`

## End-to-End Run Order

### Step 1:  dataset

Synthetic fallback:

```powershell
python src/scraper/generate_synthetic_data.py
```

Scraper scaffold:

```powershell
python src/scraper/scraper_99acres.py
```

Note: real estate websites often block automated scraping. The synthetic generator is included so the project can still be trained and demoed reliably.

### Step 2: Clean data

```powershell
python src/cleaning/clean_data.py
```

Outputs:

- `data/cleaned/pune_cleaned.csv`
- `data/cleaned/pune_cleaned_rental.csv`
- `data/cleaned/pune_cleaned_purchase.csv`

### Step 3: Feature engineering

```powershell
python src/features/feature_engineering.py
```

Outputs:

- `data/cleaned/pune_feature_engineered.csv`
- `data/cleaned/pune_feature_engineered_rental.csv`
- `data/cleaned/pune_feature_engineered_purchase.csv`

### Step 4: EDA

Script:

```powershell
python src/eda/eda_analysis.py
```

Notebook:

```powershell
jupyter notebook notebooks/EDA.ipynb
```

Generated figures are saved under `notebooks/figures/`.

### Step 5: Train models

Rental model:

```powershell
python src/models/train_rental_model.py
```

Purchase model:

```powershell
python src/models/train_purchase_model.py
```

Saved artifacts:

- `models/rental_price_model.pkl`
- `models/rental_scaler.pkl`
- `models/rental_features.pkl`
- `models/rental_metrics.json`
- `models/purchase_price_model.pkl`
- `models/purchase_scaler.pkl`
- `models/purchase_features.pkl`
- `models/purchase_metrics.json`

### Step 6: SHAP explainability

```powershell
python src/models/shap_analysis.py
```

Saved plots:

- `notebooks/figures/shap/rental_shap_bar.png`
- `notebooks/figures/shap/rental_shap_beeswarm.png`
- `notebooks/figures/shap/rental_shap_waterfall.png`
- `notebooks/figures/shap/purchase_shap_bar.png`
- `notebooks/figures/shap/purchase_shap_beeswarm.png`
- `notebooks/figures/shap/purchase_shap_waterfall.png`

### Step 7: Launch dashboard

```powershell
streamlit run src/dashboard/app.py
```

The dashboard includes:

- `Price Predictor`
  Predicts rental and purchase price with a confidence range and top SHAP drivers.
- `Market Explorer`
  Compares locality-level prices, tier trends, and distribution patterns.
- `About / Model Info`
  Shows model performance tables and feature rankings.

The app also includes a `Bootstrap Demo Assets` button to generate synthetic data and train models if artifacts are missing.

## Core Modules

- `src/scraper/scraper_99acres.py`
  99acres scraping scaffold with rate limiting, pagination, and logging.
- `src/cleaning/clean_data.py`
  Merges raw CSVs, removes sparse rows, fuzzy-deduplicates, imputes, handles outliers, and encodes features.
- `src/features/feature_engineering.py`
  Creates derived modeling features.
- `src/eda/eda_analysis.py`
  Produces the requested static EDA plots.
- `src/models/train_rental_model.py`
  Trains and saves the best rental model.
- `src/models/train_purchase_model.py`
  Trains and saves the best purchase model.
- `src/models/shap_analysis.py`
  Produces SHAP visual explanations.
- `src/dashboard/app.py`
  Streamlit dashboard for prediction, exploration, and model reporting.



