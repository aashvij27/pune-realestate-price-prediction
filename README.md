# Pune Real Estate Market Price Analysis and Prediction

An end-to-end B.Tech AIML project for analyzing Pune residential real estate and predicting two targets:

- Monthly rental price in INR/month
- Purchase price in INR

The project includes data ingestion scaffolding, listing-style sample data for local experimentation, data cleaning, EDA, feature engineering, dual-model training, SHAP explainability, and a Streamlit dashboard.

## Project Structure

```text
pune-realestate/
|-- data/
|   |-- raw/
|   |-- cleaned/
|-- models/
|-- notebooks/
|   |-- EDA.ipynb
|   |-- figures/
|-- src/
|   |-- cleaning/
|   |-- dashboard/
|   |-- eda/
|   |-- features/
|   |-- models/
|   |-- scraper/
|-- .streamlit/
|-- requirements.txt
|-- README.md
```

## Architecture

```text
Raw listing data
        |
        v
Data ingestion / scraper scaffold
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

## Dataset

The project works with structured Pune property listing data covering physical, furnishing, society, location, and market dimensions. The current repository includes listing-style sample data so the complete pipeline can be run locally without depending on live real estate websites.

The modeling datasets intentionally exclude metadata fields such as source URLs and listing-source labels. The prediction pipeline uses property attributes such as BHK, area, floors, building age, amenities, furnishing, locality, metro and IT-park distance, rent, purchase price, price per sqft, brokerage type, and listing freshness.

## Setup

1. Open a terminal in the project root.
2. Create a virtual environment:
   `python -m venv .venv`
3. Activate it on Windows PowerShell:
   `.venv\Scripts\Activate.ps1`
4. Install dependencies:
   `pip install -r requirements.txt`

## Rebuild Pipeline

Run these commands from the project root:

```powershell
python src\scraper\generate_sample_data.py
python src\cleaning\clean_data.py
python src\features\feature_engineering.py
python src\models\train_rental_model.py
python src\models\train_purchase_model.py
python src\models\shap_analysis.py
streamlit run src\dashboard\app.py
```

## Core Modules

- `src/scraper/scraper_99acres.py`  
  99acres scraping scaffold with rate limiting, pagination, and logging.
- `src/scraper/generate_sample_data.py`  
  Creates a local listing-style sample dataset for reproducible experimentation.
- `src/cleaning/clean_data.py`  
  Merges raw CSVs, removes sparse rows, fuzzy-deduplicates, imputes, handles outliers, and encodes features.
- `src/features/feature_engineering.py`  
  Creates derived modeling features.
- `src/eda/eda_analysis.py`  
  Produces static EDA plots.
- `src/models/train_rental_model.py`  
  Trains and saves the best rental model.
- `src/models/train_purchase_model.py`  
  Trains and saves the best purchase model.
- `src/models/shap_analysis.py`  
  Produces SHAP visual explanations.
- `src/dashboard/app.py`  
  Streamlit dashboard for prediction, exploration, and model reporting.
