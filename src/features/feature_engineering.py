"""Create derived features for Pune real estate listings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import CLEANED_DATA_DIR, LOCALITY_TIER_MAP, ensure_directories


def compute_connectivity_score(dataframe: pd.DataFrame) -> pd.Series:
    """Create a 0-1 connectivity score where smaller distances imply better connectivity."""

    distance_frame = dataframe[
        ["distance_to_metro_km", "distance_to_it_park_km", "distance_to_hospital_km"]
    ].copy()
    max_values = distance_frame.max().replace(0, 1)
    normalized_distance = distance_frame.divide(max_values)
    score = 1 - normalized_distance.mean(axis=1)
    return score.clip(lower=0, upper=1)


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns used in EDA and model training."""

    df = dataframe.copy()
    df["price_per_sqft"] = (
        df["purchase_price"].fillna(df["rental_price"] * 12 * 25) / df["carpet_area_sqft"].replace(0, np.nan)
    ).fillna(df.get("price_per_sqft", 0))
    df["floor_ratio"] = (
        df["floor_number"].fillna(0) / df["total_floors"].replace(0, np.nan)
    ).fillna(0)
    df["connectivity_score"] = compute_connectivity_score(df)
    df["locality_tier"] = df["locality"].map(LOCALITY_TIER_MAP).fillna(df.get("locality_tier", 3)).astype(int)
    df["area_per_bhk"] = (
        df["carpet_area_sqft"] / df["bhk_number"].replace(0, np.nan)
    ).fillna(df["carpet_area_sqft"])
    return df


def main() -> None:
    """CLI entry point for feature engineering."""

    ensure_directories()
    cleaned = pd.read_csv(CLEANED_DATA_DIR / "pune_cleaned.csv")
    engineered = add_engineered_features(cleaned)
    engineered.to_csv(CLEANED_DATA_DIR / "pune_feature_engineered.csv", index=False)
    engineered.dropna(subset=["rental_price"]).to_csv(
        CLEANED_DATA_DIR / "pune_feature_engineered_rental.csv",
        index=False,
    )
    engineered.dropna(subset=["purchase_price"]).to_csv(
        CLEANED_DATA_DIR / "pune_feature_engineered_purchase.csv",
        index=False,
    )
    print(f"Feature-engineered dataset saved to {CLEANED_DATA_DIR / 'pune_feature_engineered.csv'}")


if __name__ == "__main__":
    main()
