"""Shared constants and helpers for the Pune real estate project."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
SHAP_FIGURES_DIR = FIGURES_DIR / "shap"
MODELS_DIR = PROJECT_ROOT / "models"

LOCALITY_TIER_MAP = {
    "Koregaon Park": 1,
    "Kalyani Nagar": 1,
    "Baner": 1,
    "Kothrud": 2,
    "Aundh": 2,
    "Viman Nagar": 2,
    "Kharadi": 2,
    "Shivajinagar": 2,
    "Hadapsar": 3,
    "Nigdi": 3,
    "Pimple Saudagar": 3,
    "Wakad": 3,
    "Hinjewadi": 3,
    "Magarpatta": 3,
    "Pune Camp": 3,
}

LOCALITY_DETAILS = {
    "Koregaon Park": {"pin_code": "411001", "distance_to_metro_km": 2.5, "distance_to_it_park_km": 5.0, "distance_to_school_km": 1.5, "distance_to_hospital_km": 1.2},
    "Kalyani Nagar": {"pin_code": "411006", "distance_to_metro_km": 2.0, "distance_to_it_park_km": 4.8, "distance_to_school_km": 1.3, "distance_to_hospital_km": 1.4},
    "Baner": {"pin_code": "411045", "distance_to_metro_km": 3.1, "distance_to_it_park_km": 3.0, "distance_to_school_km": 1.8, "distance_to_hospital_km": 2.2},
    "Kothrud": {"pin_code": "411038", "distance_to_metro_km": 1.8, "distance_to_it_park_km": 7.5, "distance_to_school_km": 1.2, "distance_to_hospital_km": 1.5},
    "Aundh": {"pin_code": "411007", "distance_to_metro_km": 3.5, "distance_to_it_park_km": 4.0, "distance_to_school_km": 1.5, "distance_to_hospital_km": 1.6},
    "Viman Nagar": {"pin_code": "411014", "distance_to_metro_km": 2.4, "distance_to_it_park_km": 5.5, "distance_to_school_km": 1.7, "distance_to_hospital_km": 1.5},
    "Kharadi": {"pin_code": "411014", "distance_to_metro_km": 4.2, "distance_to_it_park_km": 1.5, "distance_to_school_km": 2.1, "distance_to_hospital_km": 2.0},
    "Shivajinagar": {"pin_code": "411005", "distance_to_metro_km": 1.0, "distance_to_it_park_km": 7.0, "distance_to_school_km": 1.0, "distance_to_hospital_km": 1.3},
    "Hadapsar": {"pin_code": "411028", "distance_to_metro_km": 5.5, "distance_to_it_park_km": 2.2, "distance_to_school_km": 2.3, "distance_to_hospital_km": 2.1},
    "Nigdi": {"pin_code": "411044", "distance_to_metro_km": 3.6, "distance_to_it_park_km": 10.5, "distance_to_school_km": 1.7, "distance_to_hospital_km": 2.4},
    "Pimple Saudagar": {"pin_code": "411027", "distance_to_metro_km": 4.1, "distance_to_it_park_km": 5.2, "distance_to_school_km": 1.5, "distance_to_hospital_km": 1.9},
    "Wakad": {"pin_code": "411057", "distance_to_metro_km": 4.8, "distance_to_it_park_km": 2.8, "distance_to_school_km": 2.0, "distance_to_hospital_km": 2.2},
    "Hinjewadi": {"pin_code": "411057", "distance_to_metro_km": 6.1, "distance_to_it_park_km": 1.0, "distance_to_school_km": 2.4, "distance_to_hospital_km": 2.6},
    "Magarpatta": {"pin_code": "411013", "distance_to_metro_km": 5.0, "distance_to_it_park_km": 1.8, "distance_to_school_km": 1.9, "distance_to_hospital_km": 2.1},
    "Pune Camp": {"pin_code": "411001", "distance_to_metro_km": 2.2, "distance_to_it_park_km": 6.4, "distance_to_school_km": 1.4, "distance_to_hospital_km": 1.8},
}

BHK_OPTIONS = ["1 RK", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"]
FURNISHING_ORDER = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
FLOORING_TYPES = ["Vitrified", "Marble", "Wooden", "Granite", "Ceramic"]
PARKING_TYPES = ["None", "Bike", "Covered", "Open", "Stilt"]
WATER_SUPPLY_TYPES = ["Municipal", "Borewell", "Corporation + Borewell", "Tanker"]
BROKERAGE_TYPES = ["Owner", "Broker", "Builder"]
SOCIETY_TYPES = ["Society", "Standalone"]

TARGET_COLUMNS = ["rental_price", "purchase_price"]
METADATA_COLUMNS = ["listing_id", "source", "listing_url"]

LISTING_SCHEMA = [
    "listing_id",
    "source",
    "listing_url",
    "bhk_type",
    "carpet_area_sqft",
    "super_built_up_area_sqft",
    "floor_number",
    "total_floors",
    "building_age_years",
    "lift_available",
    "furnishing_status",
    "modular_kitchen",
    "bathrooms",
    "balconies",
    "ac_units",
    "wardrobes",
    "flooring_type",
    "gated_community",
    "security",
    "clubhouse",
    "gym",
    "swimming_pool",
    "parking_type",
    "power_backup",
    "water_supply",
    "maintenance_charges",
    "locality",
    "pin_code",
    "society_type",
    "distance_to_metro_km",
    "distance_to_it_park_km",
    "distance_to_school_km",
    "distance_to_hospital_km",
    "rental_price",
    "purchase_price",
    "price_per_sqft",
    "brokerage_type",
    "days_since_listed",
]


def ensure_directories() -> None:
    """Create all expected output directories if they do not exist."""

    for path in [
        RAW_DATA_DIR,
        CLEANED_DATA_DIR,
        FIGURES_DIR,
        SHAP_FIGURES_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Return a configured logger for the given module name."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def bhk_to_number(value: str | int | float | None) -> int:
    """Convert BHK labels into their numeric form."""

    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper()
    if text.startswith("1 RK"):
        return 1
    digits = "".join(character for character in text if character.isdigit())
    return int(digits) if digits else 0


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Persist a dictionary as JSON with UTF-8 encoding."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

