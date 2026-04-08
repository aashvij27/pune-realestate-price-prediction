"""Streamlit dashboard for Pune real estate price prediction and market exploration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning.clean_data import add_encodings, clean_dataset, load_raw_datasets, save_cleaned_outputs
from src.common import (
    BHK_OPTIONS,
    FLOORING_TYPES,
    FURNISHING_ORDER,
    LOCALITY_DETAILS,
    LOCALITY_TIER_MAP,
    MODELS_DIR,
    PARKING_TYPES,
    RAW_DATA_DIR,
    SOCIETY_TYPES,
    WATER_SUPPLY_TYPES,
    bhk_to_number,
    ensure_directories,
)
from src.features.feature_engineering import add_engineered_features
from src.models.model_utils import prepare_features, run_training_pipeline
from src.scraper.generate_synthetic_data import generate_synthetic_dataset

st.set_page_config(
    page_title="Pune Real Estate Predictor",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f8f4ec 0%, #f3f7fb 100%);
    }
    .stApp {
        background: radial-gradient(circle at top right, rgba(217, 121, 37, 0.12), transparent 30%),
                    radial-gradient(circle at top left, rgba(39, 76, 119, 0.10), transparent 28%),
                    linear-gradient(180deg, #f8f4ec 0%, #f3f7fb 100%);
    }
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(39, 76, 119, 0.12);
        box-shadow: 0 10px 30px rgba(39, 76, 119, 0.08);
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #203040;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def bootstrap_demo_assets() -> list[str]:
    """Create the full demo data and model artifact pipeline if assets are missing."""

    ensure_directories()
    messages: list[str] = []

    raw_file = RAW_DATA_DIR / "synthetic_pune_listings.csv"
    if not raw_file.exists():
        generate_synthetic_dataset()
        messages.append("Generated synthetic Pune listings.")

    cleaned_file = PROJECT_ROOT / "data" / "cleaned" / "pune_cleaned.csv"
    if not cleaned_file.exists():
        raw_dataframe = load_raw_datasets()
        cleaned_dataframe, _ = clean_dataset(raw_dataframe)
        save_cleaned_outputs(cleaned_dataframe)
        messages.append("Cleaned raw datasets and saved encoded outputs.")

    engineered_file = PROJECT_ROOT / "data" / "cleaned" / "pune_feature_engineered.csv"
    if not engineered_file.exists():
        cleaned_dataframe = pd.read_csv(cleaned_file)
        engineered_dataframe = add_engineered_features(cleaned_dataframe)
        engineered_dataframe.to_csv(engineered_file, index=False)
        engineered_dataframe.dropna(subset=["rental_price"]).to_csv(
            PROJECT_ROOT / "data" / "cleaned" / "pune_feature_engineered_rental.csv",
            index=False,
        )
        engineered_dataframe.dropna(subset=["purchase_price"]).to_csv(
            PROJECT_ROOT / "data" / "cleaned" / "pune_feature_engineered_purchase.csv",
            index=False,
        )
        messages.append("Created feature-engineered datasets.")

    rental_model = MODELS_DIR / "rental_price_model.pkl"
    purchase_model = MODELS_DIR / "purchase_price_model.pkl"
    if not rental_model.exists():
        run_training_pipeline(
            dataset_path=PROJECT_ROOT / "data" / "cleaned" / "pune_feature_engineered_rental.csv",
            target_column="rental_price",
            model_path=MODELS_DIR / "rental_price_model.pkl",
            scaler_path=MODELS_DIR / "rental_scaler.pkl",
            features_path=MODELS_DIR / "rental_features.pkl",
            metrics_path=MODELS_DIR / "rental_metrics.json",
        )
        messages.append("Trained rental price model.")
    if not purchase_model.exists():
        run_training_pipeline(
            dataset_path=PROJECT_ROOT / "data" / "cleaned" / "pune_feature_engineered_purchase.csv",
            target_column="purchase_price",
            model_path=MODELS_DIR / "purchase_price_model.pkl",
            scaler_path=MODELS_DIR / "purchase_scaler.pkl",
            features_path=MODELS_DIR / "purchase_features.pkl",
            metrics_path=MODELS_DIR / "purchase_metrics.json",
        )
        messages.append("Trained purchase price model.")

    return messages or ["All demo assets are already available."]


@st.cache_data(show_spinner=False)
def load_market_dataset() -> pd.DataFrame:
    """Load the feature-engineered market dataset."""

    return pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "pune_feature_engineered.csv")


@st.cache_resource(show_spinner=False)
def load_artifacts(kind: str) -> tuple[object, object, list[str], dict]:
    """Load a saved model, scaler, feature list, and metrics."""

    model = joblib.load(MODELS_DIR / f"{kind}_price_model.pkl")
    scaler = joblib.load(MODELS_DIR / f"{kind}_scaler.pkl")
    features = joblib.load(MODELS_DIR / f"{kind}_features.pkl")
    with (MODELS_DIR / f"{kind}_metrics.json").open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return model, scaler, features, metrics


def build_input_row(user_inputs: dict[str, object]) -> pd.DataFrame:
    """Create a single-row dataframe aligned to the training schema."""

    locality = str(user_inputs["locality"])
    locality_details = LOCALITY_DETAILS[locality]
    base_row = {
        "listing_id": "APP-INPUT",
        "source": "dashboard",
        "listing_url": "",
        "bhk_type": user_inputs["bhk_type"],
        "carpet_area_sqft": user_inputs["carpet_area_sqft"],
        "super_built_up_area_sqft": user_inputs["super_built_up_area_sqft"],
        "floor_number": user_inputs["floor_number"],
        "total_floors": user_inputs["total_floors"],
        "building_age_years": user_inputs["building_age_years"],
        "lift_available": int(user_inputs["lift_available"]),
        "furnishing_status": user_inputs["furnishing_status"],
        "modular_kitchen": int(user_inputs["modular_kitchen"]),
        "bathrooms": user_inputs["bathrooms"],
        "balconies": user_inputs["balconies"],
        "ac_units": user_inputs["ac_units"],
        "wardrobes": user_inputs["wardrobes"],
        "flooring_type": user_inputs["flooring_type"],
        "gated_community": int(user_inputs["gated_community"]),
        "security": int(user_inputs["security"]),
        "clubhouse": int(user_inputs["clubhouse"]),
        "gym": int(user_inputs["gym"]),
        "swimming_pool": int(user_inputs["swimming_pool"]),
        "parking_type": user_inputs["parking_type"],
        "power_backup": int(user_inputs["power_backup"]),
        "water_supply": user_inputs["water_supply"],
        "maintenance_charges": user_inputs["maintenance_charges"],
        "locality": locality,
        "pin_code": locality_details["pin_code"],
        "society_type": user_inputs["society_type"],
        "distance_to_metro_km": user_inputs["distance_to_metro_km"],
        "distance_to_it_park_km": user_inputs["distance_to_it_park_km"],
        "distance_to_school_km": user_inputs["distance_to_school_km"],
        "distance_to_hospital_km": user_inputs["distance_to_hospital_km"],
        "rental_price": 0,
        "purchase_price": 0,
        "price_per_sqft": 0,
        "brokerage_type": "Owner",
        "days_since_listed": user_inputs["days_since_listed"],
    }
    input_frame = pd.DataFrame([base_row])
    input_frame = add_encodings(input_frame)
    input_frame = add_engineered_features(input_frame)
    return input_frame


def align_feature_frame(dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Align a dataframe to the saved model feature columns."""

    aligned = dataframe.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[feature_columns]


def get_prediction_and_explanation(kind: str, input_frame: pd.DataFrame, market_df: pd.DataFrame) -> tuple[float, pd.DataFrame, dict]:
    """Predict a target value and compute top SHAP contributions."""

    model, scaler, feature_columns, metrics = load_artifacts(kind)
    feature_frame = align_feature_frame(input_frame, feature_columns)
    scaled_row = pd.DataFrame(
        scaler.transform(feature_frame),
        columns=feature_columns,
        index=feature_frame.index,
    )

    market_features, _ = prepare_features(
        market_df.dropna(subset=[f"{kind}_price"]),
        target_column=f"{kind}_price",
    )
    market_features = align_feature_frame(market_features, feature_columns)
    background = pd.DataFrame(
        scaler.transform(market_features.sample(min(250, len(market_features)), random_state=42)),
        columns=feature_columns,
    )

    prediction = float(model.predict(scaled_row)[0])
    explainer = shap.Explainer(model, background)
    shap_values = explainer(scaled_row, check_additivity=False)

    contribution_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "shap_value": shap_values.values[0],
        }
    )
    contribution_frame["abs_shap"] = contribution_frame["shap_value"].abs()
    contribution_frame = contribution_frame.sort_values("abs_shap", ascending=False).head(5)
    return prediction, contribution_frame, metrics


def render_metric_card(title: str, value: str, help_text: str) -> None:
    """Display a styled metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.9rem;color:#5b6b7f;">{title}</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1f3347;">{value}</div>
            <div style="font-size:0.9rem;color:#708090;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Pune Real Estate Market Price Analysis and Prediction")
st.caption("Dual-model prediction system for Pune residential rental and purchase prices.")

if st.button("Bootstrap Demo Assets"):
    with st.spinner("Preparing synthetic data, cleaned datasets, and trained models..."):
        for message in bootstrap_demo_assets():
            st.success(message)
    st.cache_data.clear()
    st.cache_resource.clear()

required_artifacts = [
    PROJECT_ROOT / "data" / "cleaned" / "pune_feature_engineered.csv",
    MODELS_DIR / "rental_price_model.pkl",
    MODELS_DIR / "purchase_price_model.pkl",
]
if not all(path.exists() for path in required_artifacts):
    st.warning(
        "Model artifacts are missing. Click 'Bootstrap Demo Assets' to generate synthetic data, train both models, and unlock the dashboard."
    )
    st.stop()

market_df = load_market_dataset()

page = st.sidebar.radio("Navigate", ["Price Predictor", "Market Explorer", "About / Model Info"])

if page == "Price Predictor":
    st.sidebar.header("Property Features")
    locality = st.sidebar.selectbox("Locality", list(LOCALITY_TIER_MAP.keys()))
    bhk_type = st.sidebar.selectbox("BHK Type", BHK_OPTIONS, index=2)
    carpet_area_sqft = st.sidebar.slider("Carpet Area (sqft)", 250, 2500, 950, 25)
    super_built_up_area_sqft = st.sidebar.slider("Super Built-up Area (sqft)", 300, 3200, 1180, 25)
    furnishing_status = st.sidebar.selectbox("Furnishing Status", FURNISHING_ORDER, index=1)
    floor_number = st.sidebar.slider("Floor Number", 0, 35, 5)
    total_floors = st.sidebar.slider("Total Floors", 1, 40, 12)
    building_age_years = st.sidebar.slider("Building Age (years)", 0, 30, 6)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 6, max(1, min(4, bhk_to_number(bhk_type))))
    balconies = st.sidebar.slider("Balconies", 0, 5, 2)
    ac_units = st.sidebar.slider("AC Units", 0, 6, max(0, bhk_to_number(bhk_type) - 1))
    wardrobes = st.sidebar.slider("Wardrobes", 1, 10, max(2, bhk_to_number(bhk_type) * 2))
    maintenance_charges = st.sidebar.slider("Maintenance Charges (INR)", 500, 15000, 3500, 100)
    flooring_type = st.sidebar.selectbox("Flooring Type", FLOORING_TYPES)
    parking_type = st.sidebar.selectbox("Parking Type", PARKING_TYPES, index=2)
    water_supply = st.sidebar.selectbox("Water Supply", WATER_SUPPLY_TYPES)
    society_type = st.sidebar.selectbox("Society Type", SOCIETY_TYPES)
    distance_to_metro_km = st.sidebar.slider("Distance to Metro (km)", 0.2, 10.0, float(LOCALITY_DETAILS[locality]["distance_to_metro_km"]), 0.1)
    distance_to_it_park_km = st.sidebar.slider("Distance to IT Park (km)", 0.2, 15.0, float(LOCALITY_DETAILS[locality]["distance_to_it_park_km"]), 0.1)
    distance_to_school_km = st.sidebar.slider("Distance to School (km)", 0.2, 8.0, float(LOCALITY_DETAILS[locality]["distance_to_school_km"]), 0.1)
    distance_to_hospital_km = st.sidebar.slider("Distance to Hospital (km)", 0.2, 8.0, float(LOCALITY_DETAILS[locality]["distance_to_hospital_km"]), 0.1)
    days_since_listed = st.sidebar.slider("Days Since Listed", 1, 120, 18)
    lift_available = st.sidebar.checkbox("Lift Available", value=True)
    modular_kitchen = st.sidebar.checkbox("Modular Kitchen", value=True)
    gated_community = st.sidebar.checkbox("Gated Community", value=True)
    security = st.sidebar.checkbox("Security", value=True)
    clubhouse = st.sidebar.checkbox("Clubhouse", value=True)
    gym = st.sidebar.checkbox("Gym", value=True)
    swimming_pool = st.sidebar.checkbox("Swimming Pool", value=False)
    power_backup = st.sidebar.checkbox("Power Backup", value=True)

    user_inputs = {
        "locality": locality,
        "bhk_type": bhk_type,
        "carpet_area_sqft": carpet_area_sqft,
        "super_built_up_area_sqft": super_built_up_area_sqft,
        "furnishing_status": furnishing_status,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "building_age_years": building_age_years,
        "bathrooms": bathrooms,
        "balconies": balconies,
        "ac_units": ac_units,
        "wardrobes": wardrobes,
        "maintenance_charges": maintenance_charges,
        "flooring_type": flooring_type,
        "parking_type": parking_type,
        "water_supply": water_supply,
        "society_type": society_type,
        "distance_to_metro_km": distance_to_metro_km,
        "distance_to_it_park_km": distance_to_it_park_km,
        "distance_to_school_km": distance_to_school_km,
        "distance_to_hospital_km": distance_to_hospital_km,
        "days_since_listed": days_since_listed,
        "lift_available": lift_available,
        "modular_kitchen": modular_kitchen,
        "gated_community": gated_community,
        "security": security,
        "clubhouse": clubhouse,
        "gym": gym,
        "swimming_pool": swimming_pool,
        "power_backup": power_backup,
    }

    input_frame = build_input_row(user_inputs)
    rental_prediction, rental_shap, rental_metrics = get_prediction_and_explanation("rental", input_frame, market_df)
    purchase_prediction, purchase_shap, purchase_metrics = get_prediction_and_explanation("purchase", input_frame, market_df)

    locality_avg_rent = market_df.groupby("locality")["rental_price"].mean().get(locality, rental_prediction)
    locality_avg_purchase = market_df.groupby("locality")["purchase_price"].mean().get(locality, purchase_prediction)

    col1, col2 = st.columns(2)
    with col1:
        render_metric_card(
            "Rental Price Prediction",
            f"INR {rental_prediction:,.0f}/month",
            f"Confidence range: INR {rental_prediction * 0.9:,.0f} to {rental_prediction * 1.1:,.0f}",
        )
        st.info(
            f"This property is {'above' if rental_prediction > locality_avg_rent else 'below'} the {locality} average rental benchmark of INR {locality_avg_rent:,.0f}/month."
        )
    with col2:
        render_metric_card(
            "Purchase Price Prediction",
            f"INR {purchase_prediction:,.0f}",
            f"Confidence range: INR {purchase_prediction * 0.9:,.0f} to {purchase_prediction * 1.1:,.0f}",
        )
        st.info(
            f"This property is {'above' if purchase_prediction > locality_avg_purchase else 'below'} the {locality} average purchase benchmark of INR {locality_avg_purchase:,.0f}."
        )

    shap_col1, shap_col2 = st.columns(2)
    with shap_col1:
        st.markdown('<div class="section-header">Top Rental Drivers</div>', unsafe_allow_html=True)
        rental_chart = px.bar(
            rental_shap.sort_values("shap_value"),
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="Tealrose",
        )
        rental_chart.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(rental_chart, use_container_width=True)
    with shap_col2:
        st.markdown('<div class="section-header">Top Purchase Drivers</div>', unsafe_allow_html=True)
        purchase_chart = px.bar(
            purchase_shap.sort_values("shap_value"),
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="balance",
        )
        purchase_chart.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(purchase_chart, use_container_width=True)

elif page == "Market Explorer":
    st.sidebar.header("Explorer Filters")
    bhk_filter = st.sidebar.multiselect("BHK Types", sorted(market_df["bhk_type"].dropna().unique()), default=sorted(market_df["bhk_type"].dropna().unique()))
    listing_type = st.sidebar.radio("Listing Type", ["rental", "purchase"])
    filtered_df = market_df[market_df["bhk_type"].isin(bhk_filter)].copy()
    target_column = "rental_price" if listing_type == "rental" else "purchase_price"

    st.markdown('<div class="section-header">Average Prices by Locality</div>', unsafe_allow_html=True)
    locality_avg = filtered_df.groupby("locality", as_index=False)[target_column].mean().sort_values(target_column, ascending=False)
    st.plotly_chart(
        px.bar(
            locality_avg,
            x="locality",
            y=target_column,
            color="locality",
            title=f"Average {'Rental' if listing_type == 'rental' else 'Purchase'} Prices by Locality",
        ).update_layout(showlegend=False, xaxis_title="", yaxis_title="Price"),
        use_container_width=True,
    )

    tier_summary = filtered_df.groupby("locality_tier", as_index=False)[["rental_price", "purchase_price"]].mean()
    tier_summary = tier_summary.melt(id_vars="locality_tier", var_name="price_type", value_name="price")
    st.markdown('<div class="section-header">Price Trend Comparison Across Tiers</div>', unsafe_allow_html=True)
    st.plotly_chart(
        px.line(
            tier_summary,
            x="locality_tier",
            y="price",
            color="price_type",
            markers=True,
            title="Average Prices Across Locality Tiers",
        ),
        use_container_width=True,
    )

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.plotly_chart(
            px.box(
                filtered_df,
                x="bhk_type",
                y=target_column,
                color="furnishing_status",
                title=f"{'Rental' if listing_type == 'rental' else 'Purchase'} Distribution by BHK and Furnishing",
            ),
            use_container_width=True,
        )
    with dist_col2:
        st.plotly_chart(
            px.scatter(
                filtered_df,
                x="carpet_area_sqft",
                y=target_column,
                color="locality_tier",
                hover_data=["locality", "bhk_type"],
                title="Area vs Price Relationship",
            ),
            use_container_width=True,
        )

else:
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    rental_model, _, _, rental_metrics = load_artifacts("rental")
    purchase_model, _, _, purchase_metrics = load_artifacts("purchase")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.subheader("Rental Model")
        st.dataframe(pd.DataFrame(rental_metrics["comparison_table"]), use_container_width=True)
    with metric_col2:
        st.subheader("Purchase Model")
        st.dataframe(pd.DataFrame(purchase_metrics["comparison_table"]), use_container_width=True)

    st.markdown('<div class="section-header">Feature Importance Rankings</div>', unsafe_allow_html=True)
    importance_col1, importance_col2 = st.columns(2)
    with importance_col1:
        st.write("Rental top features")
        st.dataframe(pd.DataFrame(rental_metrics["top_features"]), use_container_width=True)
    with importance_col2:
        st.write("Purchase top features")
        st.dataframe(pd.DataFrame(purchase_metrics["top_features"]), use_container_width=True)

    st.markdown('<div class="section-header">Project Summary</div>', unsafe_allow_html=True)
    st.write(
        """
        This dashboard demonstrates an end-to-end AIML project for Pune residential real estate.
        It supports synthetic-data bootstrapping, separate rental and purchase price prediction models,
        SHAP-based explainability, and interactive market exploration across locality tiers.
        """
    )
