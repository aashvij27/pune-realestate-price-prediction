"""Train the rental price prediction model suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import CLEANED_DATA_DIR, MODELS_DIR
from src.models.model_utils import (
    evaluate_models,
    prepare_features,
    run_training_pipeline,
    save_training_artifacts,
    split_and_scale,
    top_feature_importances,
)


def main() -> None:
    """CLI entry point for rental model training."""

    comparison, _ = run_training_pipeline(
        dataset_path=CLEANED_DATA_DIR / "pune_feature_engineered_rental.csv",
        target_column="rental_price",
        model_path=MODELS_DIR / "rental_price_model.pkl",
        scaler_path=MODELS_DIR / "rental_scaler.pkl",
        features_path=MODELS_DIR / "rental_features.pkl",
        metrics_path=MODELS_DIR / "rental_metrics.json",
    )

    print("Rental Model Comparison")
    print(comparison.to_string(index=False))
    print(f"\nBest rental model saved as {MODELS_DIR / 'rental_price_model.pkl'}")


if __name__ == "__main__":
    main()
