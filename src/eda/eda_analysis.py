"""Generate exploratory data analysis plots for the Pune real estate dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common import CLEANED_DATA_DIR, FIGURES_DIR, ensure_directories

sns.set_theme(style="whitegrid")


def save_figure(output_path: Path) -> None:
    """Save the current Matplotlib figure and close it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_eda_plots(dataframe: pd.DataFrame) -> None:
    """Generate and save the required EDA visualizations."""

    ensure_directories()
    figures_dir = FIGURES_DIR

    plt.figure(figsize=(10, 6))
    median_rent = dataframe["rental_price"].median()
    sns.histplot(dataframe["rental_price"], kde=True, bins=40, color="#D97925")
    plt.axvline(median_rent, color="#274C77", linestyle="--", label=f"Median: {median_rent:,.0f}")
    plt.title("Rental Price Distribution")
    plt.xlabel("Rental Price (INR/month)")
    plt.legend()
    save_figure(figures_dir / "rental_price_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe["purchase_price"], kde=True, bins=45, color="#274C77")
    plt.title("Purchase Price Distribution")
    plt.xlabel("Purchase Price (INR)")
    save_figure(figures_dir / "purchase_price_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe, x="locality_tier", y="purchase_price", hue="locality_tier", palette="Set2", legend=False)
    plt.title("Purchase Price by Locality Tier")
    plt.xlabel("Locality Tier")
    save_figure(figures_dir / "purchase_price_by_tier.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe, x="bhk_type", y="rental_price", hue="bhk_type", palette="Set3", legend=False)
    plt.title("Rental Price by BHK Type")
    plt.xlabel("BHK Type")
    save_figure(figures_dir / "rental_price_by_bhk.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=dataframe,
        x="furnishing_status",
        y="purchase_price",
        hue="furnishing_status",
        palette="Pastel1",
        legend=False,
    )
    plt.title("Purchase Price by Furnishing Status")
    plt.xlabel("Furnishing Status")
    save_figure(figures_dir / "purchase_price_by_furnishing.png")

    numerical_columns = dataframe.select_dtypes(include="number").columns
    plt.figure(figsize=(15, 10))
    sns.heatmap(dataframe[numerical_columns].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    save_figure(figures_dir / "correlation_heatmap.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=dataframe,
        x="carpet_area_sqft",
        y="rental_price",
        hue="locality_tier",
        palette="viridis",
        alpha=0.8,
    )
    plt.title("Carpet Area vs Rental Price")
    plt.xlabel("Carpet Area (sqft)")
    plt.ylabel("Rental Price (INR/month)")
    save_figure(figures_dir / "carpet_area_vs_rental.png")

    locality_avg = (
        dataframe.groupby("locality", as_index=False)["price_per_sqft"]
        .mean()
        .sort_values("price_per_sqft", ascending=False)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=locality_avg, x="locality", y="price_per_sqft", hue="locality", palette="flare", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Price per Sqft by Locality")
    plt.xlabel("Locality")
    plt.ylabel("Average Price per Sqft")
    save_figure(figures_dir / "average_price_per_sqft_by_locality.png")


def main() -> None:
    """CLI entry point for EDA plotting."""

    ensure_directories()
    dataframe = pd.read_csv(CLEANED_DATA_DIR / "pune_feature_engineered.csv")
    generate_eda_plots(dataframe)
    print(f"EDA plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
