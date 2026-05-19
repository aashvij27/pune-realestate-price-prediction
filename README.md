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



