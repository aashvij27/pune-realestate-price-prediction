"""Clean and encode Pune real estate listing data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.preprocessing import MinMaxScaler

from src.common import (
    CLEANED_DATA_DIR,
    FURNISHING_ORDER,
    LISTING_SCHEMA,
    LOCALITY_TIER_MAP,
    METADATA_COLUMNS,
    RAW_DATA_DIR,
    bhk_to_number,
    ensure_directories,
    get_logger,
)

FUZZY_THRESHOLD = 85


def load_raw_datasets(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and merge all raw CSV files in the raw data directory."""

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    dataframes = []
    for file_path in csv_files:
        dataframe = pd.read_csv(file_path)
        missing_columns = [column for column in LISTING_SCHEMA if column not in dataframe.columns]
        for column in missing_columns:
            dataframe[column] = np.nan
        dataframe = dataframe[LISTING_SCHEMA]
        dataframes.append(dataframe)
    return pd.concat(dataframes, ignore_index=True)


def drop_sparse_rows(dataframe: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """Drop listings that are missing more than the provided fraction of features."""

    minimum_non_null = int(len(dataframe.columns) * (1 - threshold))
    return dataframe.dropna(thresh=minimum_non_null).reset_index(drop=True)


def deduplicate_listings(dataframe: pd.DataFrame, threshold: int = FUZZY_THRESHOLD) -> tuple[pd.DataFrame, int]:
    """Deduplicate near-identical listings using fuzzy matching on locality, BHK, area, and price."""

    df = dataframe.copy()
    df["comparison_key"] = (
        df["locality"].fillna("unknown").astype(str).str.lower()
        + " | "
        + df["bhk_type"].fillna("unknown").astype(str).str.lower()
        + " | "
        + df["carpet_area_sqft"].fillna(0).round(0).astype(int).astype(str)
        + " | "
        + df[["rental_price", "purchase_price"]]
        .fillna(0)
        .max(axis=1)
        .round(-3)
        .astype(int)
        .astype(str)
    )

    kept_indices: list[int] = []
    seen_keys: list[str] = []
    for index, key in df["comparison_key"].items():
        if any(fuzz.token_sort_ratio(key, existing) >= threshold for existing in seen_keys[-200:]):
            continue
        kept_indices.append(index)
        seen_keys.append(key)

    deduplicated = df.loc[kept_indices].drop(columns=["comparison_key"]).reset_index(drop=True)
    return deduplicated, len(df) - len(deduplicated)


def impute_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Impute numerical values locality-wise and categorical values with a default label."""

    df = dataframe.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    for column in numeric_columns:
        if column in {"rental_price", "purchase_price"}:
            continue
        locality_median = df.groupby("locality")[column].transform("median")
        global_median = df[column].median()
        df[column] = df[column].fillna(locality_median).fillna(global_median)

    for column in categorical_columns:
        df[column] = df[column].fillna("Not Specified")

    return df


def remove_outliers_iqr(dataframe: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Remove outliers using an IQR rule for the supplied target columns."""

    mask = pd.Series(True, index=dataframe.index)
    for column in columns:
        valid = dataframe[column].dropna()
        if valid.empty:
            continue
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask &= dataframe[column].between(lower_bound, upper_bound, inclusive="both")

    filtered = dataframe.loc[mask].reset_index(drop=True)
    return filtered, len(dataframe) - len(filtered)


def add_encodings(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add ordinal and one-hot encoded columns without dropping original fields."""

    df = dataframe.copy()
    furnishing_map = {name: index for index, name in enumerate(FURNISHING_ORDER)}
    df["furnishing_status_encoded"] = df["furnishing_status"].map(furnishing_map).fillna(-1).astype(int)
    df["bhk_number"] = df["bhk_type"].apply(bhk_to_number)
    df["bhk_type_encoded"] = df["bhk_number"]
    locality_dummies = pd.get_dummies(df["locality"], prefix="locality", dtype=int)
    df = pd.concat([df, locality_dummies], axis=1)
    return df


def add_normalized_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add min-max normalized copies of numerical columns for analysis."""

    df = dataframe.copy()
    scaler = MinMaxScaler()
    numeric_columns = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in {"rental_price", "purchase_price", "furnishing_status_encoded", "bhk_number", "bhk_type_encoded"}
    ]
    if numeric_columns:
        scaled_values = scaler.fit_transform(df[numeric_columns])
        normalized = pd.DataFrame(
            scaled_values,
            columns=[f"norm_{column}" for column in numeric_columns],
            index=df.index,
        )
        df = pd.concat([df, normalized], axis=1)
    return df


def clean_dataset(input_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the full cleaning pipeline to a merged raw dataset."""

    summary = {"rows_loaded": len(input_dataframe)}
    dataframe = drop_sparse_rows(input_dataframe)
    summary["rows_after_sparse_drop"] = len(dataframe)

    dataframe, duplicates_removed = deduplicate_listings(dataframe)
    summary["duplicates_removed"] = duplicates_removed

    dataframe = impute_values(dataframe)

    dataframe["locality_tier"] = dataframe["locality"].map(LOCALITY_TIER_MAP).fillna(3).astype(int)
    dataframe, outliers_removed = remove_outliers_iqr(dataframe, ["rental_price", "purchase_price"])
    summary["outliers_removed"] = outliers_removed

    dataframe = add_encodings(dataframe)
    dataframe = dataframe.drop(columns=METADATA_COLUMNS, errors="ignore")
    dataframe = add_normalized_columns(dataframe)
    summary["rows_final"] = len(dataframe)
    return dataframe, summary


def save_cleaned_outputs(dataframe: pd.DataFrame) -> None:
    """Save the cleaned master dataset and target-specific subsets."""

    ensure_directories()
    modeling_dataframe = dataframe.drop(columns=METADATA_COLUMNS, errors="ignore")
    modeling_dataframe.to_csv(CLEANED_DATA_DIR / "pune_cleaned.csv", index=False)
    modeling_dataframe.dropna(subset=["rental_price"]).to_csv(CLEANED_DATA_DIR / "pune_cleaned_rental.csv", index=False)
    modeling_dataframe.dropna(subset=["purchase_price"]).to_csv(CLEANED_DATA_DIR / "pune_cleaned_purchase.csv", index=False)


def main() -> None:
    """CLI entry point for the cleaning pipeline."""

    logger = get_logger("clean_data", CLEANED_DATA_DIR / "cleaning.log")
    raw_dataframe = load_raw_datasets()
    cleaned_dataframe, summary = clean_dataset(raw_dataframe)
    save_cleaned_outputs(cleaned_dataframe)

    logger.info("Cleaning summary: %s", summary)
    print("Cleaning Summary Report")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
