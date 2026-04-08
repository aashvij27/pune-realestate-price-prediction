"""Shared utilities for training and evaluating price prediction models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.common import MODELS_DIR, RANDOM_STATE, save_json

NON_FEATURE_COLUMNS = {
    "listing_id",
    "source",
    "listing_url",
    "bhk_type",
    "furnishing_status",
    "flooring_type",
    "parking_type",
    "water_supply",
    "locality",
    "pin_code",
    "society_type",
    "brokerage_type",
}


def prepare_features(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a numeric feature matrix and target vector."""

    features = dataframe.drop(columns=[column for column in ["rental_price", "purchase_price"] if column in dataframe.columns and column != target_column])
    features = features.drop(columns=[column for column in NON_FEATURE_COLUMNS if column in features.columns], errors="ignore")
    features = features.drop(columns=[column for column in features.columns if column.startswith("norm_")], errors="ignore")
    features = features.select_dtypes(include="number").copy()
    features = features.drop(columns=[target_column], errors="ignore")
    target = dataframe[target_column].astype(float)
    return features, target


def split_and_scale(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Create a train-test split and fit a standard scaler."""

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def build_model_suite() -> dict[str, object]:
    """Create the candidate regressor suite."""

    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LGBMRegressor": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Cross-validate and evaluate all candidate models."""

    results = []
    fitted_models: dict[str, object] = {}
    cross_validator = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    for model_name, model in build_model_suite().items():
        cv_scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        result = {
            "model": model_name,
            "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
            "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
            "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "test_mae": float(mean_absolute_error(y_test, predictions)),
            "test_r2": float(r2_score(y_test, predictions)),
        }
        results.append(result)
        fitted_models[model_name] = model

    comparison = pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)
    return comparison, fitted_models


def save_training_artifacts(
    model: object,
    scaler: StandardScaler,
    feature_columns: list[str],
    metrics: dict[str, object],
    model_path: Path,
    scaler_path: Path,
    features_path: Path,
    metrics_path: Path,
) -> None:
    """Save trained model assets to disk."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_columns, features_path)
    save_json(metrics, metrics_path)


def top_feature_importances(model: object, feature_columns: list[str], top_n: int = 15) -> list[dict[str, float]]:
    """Compute model feature importance rankings with best-effort fallbacks."""

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_))
    else:
        importances = np.zeros(len(feature_columns))

    ranking = pd.DataFrame({"feature": feature_columns, "importance": importances})
    ranking = ranking.sort_values("importance", ascending=False).head(top_n)
    return ranking.to_dict(orient="records")


def run_training_pipeline(
    dataset_path: Path,
    target_column: str,
    model_path: Path,
    scaler_path: Path,
    features_path: Path,
    metrics_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train the full model suite and save the best artifact set."""

    dataset = pd.read_csv(dataset_path)
    features, target = prepare_features(dataset, target_column=target_column)
    x_train, x_test, y_train, y_test, scaler = split_and_scale(features, target)
    comparison, fitted_models = evaluate_models(x_train, x_test, y_train, y_test)
    best_model_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    metrics = {
        "target": target_column,
        "best_model": best_model_name,
        "comparison_table": comparison.round(4).to_dict(orient="records"),
        "top_features": top_feature_importances(best_model, features.columns.tolist()),
    }
    save_training_artifacts(
        model=best_model,
        scaler=scaler,
        feature_columns=features.columns.tolist(),
        metrics=metrics,
        model_path=model_path,
        scaler_path=scaler_path,
        features_path=features_path,
        metrics_path=metrics_path,
    )
    return comparison, metrics
