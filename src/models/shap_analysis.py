"""Generate SHAP explainability artifacts for the best trained models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.common import CLEANED_DATA_DIR, MODELS_DIR, SHAP_FIGURES_DIR, ensure_directories, get_logger, save_json
from src.models.model_utils import prepare_features


def generate_shap_outputs(
    dataset_path,
    target_column: str,
    model_path,
    scaler_path,
    feature_path,
    prefix: str,
) -> list[dict[str, float]]:
    """Create SHAP plots and return the top 10 feature importance rows."""

    dataset = pd.read_csv(dataset_path)
    features, _ = prepare_features(dataset, target_column=target_column)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_path)

    feature_frame = features[feature_columns]
    scaled_values = pd.DataFrame(
        scaler.transform(feature_frame),
        columns=feature_columns,
        index=feature_frame.index,
    )
    explainer = shap.Explainer(model, scaled_values)
    shap_values = explainer(scaled_values, check_additivity=False)

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(SHAP_FIGURES_DIR / f"{prefix}_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values.values, feature_frame, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(SHAP_FIGURES_DIR / f"{prefix}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.waterfall(shap_values[0], show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(SHAP_FIGURES_DIR / f"{prefix}_shap_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("importance", ascending=False)
    return importance.head(10).to_dict(orient="records")


def main() -> None:
    """CLI entry point for SHAP analysis."""

    ensure_directories()
    logger = get_logger("shap_analysis", SHAP_FIGURES_DIR / "shap_analysis.log")

    rental_top = generate_shap_outputs(
        dataset_path=CLEANED_DATA_DIR / "pune_feature_engineered_rental.csv",
        target_column="rental_price",
        model_path=MODELS_DIR / "rental_price_model.pkl",
        scaler_path=MODELS_DIR / "rental_scaler.pkl",
        feature_path=MODELS_DIR / "rental_features.pkl",
        prefix="rental",
    )
    purchase_top = generate_shap_outputs(
        dataset_path=CLEANED_DATA_DIR / "pune_feature_engineered_purchase.csv",
        target_column="purchase_price",
        model_path=MODELS_DIR / "purchase_price_model.pkl",
        scaler_path=MODELS_DIR / "purchase_scaler.pkl",
        feature_path=MODELS_DIR / "purchase_features.pkl",
        prefix="purchase",
    )

    summary = {
        "rental_top_features": rental_top,
        "purchase_top_features": purchase_top,
    }
    save_json(summary, MODELS_DIR / "shap_top_features.json")

    logger.info("Top rental features: %s", json.dumps(rental_top))
    logger.info("Top purchase features: %s", json.dumps(purchase_top))
    print("Top 10 rental price features")
    for row in rental_top:
        print(f"- {row['feature']}: {row['importance']:.4f}")
    print("\nTop 10 purchase price features")
    for row in purchase_top:
        print(f"- {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
