"""Generate a realistic Pune real estate sample dataset for demos and model training."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.common import (
    BHK_OPTIONS,
    BROKERAGE_TYPES,
    FLOORING_TYPES,
    FURNISHING_ORDER,
    LISTING_SCHEMA,
    LOCALITY_DETAILS,
    LOCALITY_TIER_MAP,
    PARKING_TYPES,
    RAW_DATA_DIR,
    RANDOM_STATE,
    SOCIETY_TYPES,
    WATER_SUPPLY_TYPES,
    bhk_to_number,
    ensure_directories,
    get_logger,
)


@dataclass(frozen=True)
class TierProfile:
    """Pricing and amenity assumptions for a locality tier."""

    rent_rate_range: tuple[float, float]
    affordable_price_range: tuple[float, float]
    premium_price_range: tuple[float, float]
    premium_probability: float


TIER_PROFILES = {
    1: TierProfile((38.0, 70.0), (9_500, 15_500), (19_000, 28_000), 0.72),
    2: TierProfile((25.0, 48.0), (7_500, 10_500), (12_000, 17_500), 0.33),
    3: TierProfile((15.0, 34.0), (5_800, 8_200), (9_000, 13_000), 0.18),
}


def weighted_choice(rng: np.random.Generator, values: list[str], weights: list[float]) -> str:
    """Return a weighted random value from a list."""

    return str(rng.choice(values, p=np.array(weights) / np.sum(weights)))


def create_listing(locality: str, index: int, rng: np.random.Generator) -> dict[str, object]:
    """Create one sample listing with realistic locality-tier pricing logic."""

    tier = LOCALITY_TIER_MAP[locality]
    locality_details = LOCALITY_DETAILS[locality]
    tier_profile = TIER_PROFILES[tier]

    bhk_type = weighted_choice(
        rng,
        BHK_OPTIONS,
        [0.05, 0.22, 0.38, 0.23, 0.10, 0.02],
    )
    bhk_number = bhk_to_number(bhk_type)

    carpet_area = max(
        260,
        rng.normal(
            loc={1: 420, 2: 620, 3: 980, 4: 1420, 5: 1850}.get(bhk_number, 620),
            scale={1: 60, 2: 120, 3: 180, 4: 220, 5: 280}.get(bhk_number, 140),
        ),
    )
    super_built_up = carpet_area * rng.uniform(1.08, 1.28)
    total_floors = int(max(4, rng.normal(loc=9 + tier * 2, scale=3)))
    floor_number = int(rng.integers(0, total_floors + 1))
    building_age = int(max(0, rng.normal(loc=7 + tier, scale=4)))

    furnishing_status = weighted_choice(
        rng,
        FURNISHING_ORDER,
        {1: [0.10, 0.45, 0.45], 2: [0.22, 0.55, 0.23], 3: [0.40, 0.47, 0.13]}[tier],
    )
    society_type = weighted_choice(rng, SOCIETY_TYPES, [0.75, 0.25])
    flooring_type = weighted_choice(rng, FLOORING_TYPES, [0.38, 0.17, 0.14, 0.16, 0.15])
    parking_type = weighted_choice(rng, PARKING_TYPES, [0.06, 0.10, 0.55, 0.17, 0.12])
    water_supply = weighted_choice(rng, WATER_SUPPLY_TYPES, [0.42, 0.16, 0.27, 0.15])
    brokerage_type = weighted_choice(rng, BROKERAGE_TYPES, [0.48, 0.42, 0.10])

    bathrooms = max(1, min(6, int(round(bhk_number + rng.normal(loc=0.35, scale=0.45)))))
    balconies = max(0, min(5, int(round(bhk_number - 1 + rng.normal(loc=0.8, scale=0.7)))))
    ac_units = max(0, int(round((bhk_number - 1) * rng.uniform(0.4, 1.2) + (tier == 1))))
    wardrobes = max(1, int(round(bhk_number * rng.uniform(1.0, 2.0))))

    lift_available = int(total_floors >= 5 or rng.random() < 0.7)
    modular_kitchen = int(rng.random() < (0.92 if furnishing_status != "Unfurnished" else 0.45))
    gated_community = int(rng.random() < (0.87 if society_type == "Society" else 0.22))
    security = int(rng.random() < (0.90 if gated_community else 0.35))
    clubhouse = int(rng.random() < {1: 0.72, 2: 0.46, 3: 0.20}[tier])
    gym = int(rng.random() < {1: 0.78, 2: 0.55, 3: 0.28}[tier])
    swimming_pool = int(rng.random() < {1: 0.66, 2: 0.34, 3: 0.11}[tier])
    power_backup = int(rng.random() < {1: 0.82, 2: 0.64, 3: 0.39}[tier])

    maintenance = max(
        800,
        rng.normal(
            loc=carpet_area * {1: 4.8, 2: 3.7, 3: 2.7}[tier],
            scale=650,
        ),
    )

    distance_to_metro = max(0.2, locality_details["distance_to_metro_km"] + rng.normal(0, 0.5))
    distance_to_it_park = max(0.2, locality_details["distance_to_it_park_km"] + rng.normal(0, 0.7))
    distance_to_school = max(0.2, locality_details["distance_to_school_km"] + rng.normal(0, 0.3))
    distance_to_hospital = max(0.2, locality_details["distance_to_hospital_km"] + rng.normal(0, 0.25))
    days_since_listed = int(np.clip(rng.gamma(shape=2.3, scale=12), 1, 120))

    furnishing_multiplier = {"Unfurnished": 0.93, "Semi-Furnished": 1.00, "Fully Furnished": 1.14}[furnishing_status]
    amenity_multiplier = (
        1
        + gym * 0.02
        + swimming_pool * 0.025
        + clubhouse * 0.02
        + gated_community * 0.03
        + security * 0.015
        + lift_available * 0.015
        + power_backup * 0.02
        + modular_kitchen * 0.015
    )

    premium_segment = rng.random() < tier_profile.premium_probability
    per_sqft_rate_range = (
        tier_profile.premium_price_range if premium_segment else tier_profile.affordable_price_range
    )
    purchase_rate_per_sqft = rng.uniform(*per_sqft_rate_range)
    rent_rate_per_sqft = rng.uniform(*tier_profile.rent_rate_range)

    floor_multiplier = 1 + min(floor_number, 12) * 0.004
    age_multiplier = max(0.82, 1 - building_age * 0.01)
    connectivity_multiplier = 1 + max(0, (7 - distance_to_it_park - distance_to_metro) * 0.02)

    purchase_price = (
        carpet_area
        * purchase_rate_per_sqft
        * furnishing_multiplier
        * amenity_multiplier
        * floor_multiplier
        * age_multiplier
        * connectivity_multiplier
        + maintenance * 18
        + rng.normal(0, 350_000)
    )
    rental_price = (
        carpet_area
        * rent_rate_per_sqft
        * furnishing_multiplier
        * amenity_multiplier
        * floor_multiplier
        * max(0.85, age_multiplier)
        * connectivity_multiplier
        + maintenance * 0.25
        + rng.normal(0, 2_500)
    )

    purchase_price = float(max(2_800_000, purchase_price))
    rental_price = float(max(7_500, rental_price))

    return {
        "listing_id": f"PUN-{index:05d}",
        "source": "sample",
        "listing_url": "",
        "bhk_type": bhk_type,
        "carpet_area_sqft": round(carpet_area, 2),
        "super_built_up_area_sqft": round(super_built_up, 2),
        "floor_number": floor_number,
        "total_floors": total_floors,
        "building_age_years": building_age,
        "lift_available": lift_available,
        "furnishing_status": furnishing_status,
        "modular_kitchen": modular_kitchen,
        "bathrooms": bathrooms,
        "balconies": balconies,
        "ac_units": ac_units,
        "wardrobes": wardrobes,
        "flooring_type": flooring_type,
        "gated_community": gated_community,
        "security": security,
        "clubhouse": clubhouse,
        "gym": gym,
        "swimming_pool": swimming_pool,
        "parking_type": parking_type,
        "power_backup": power_backup,
        "water_supply": water_supply,
        "maintenance_charges": round(maintenance, 2),
        "locality": locality,
        "pin_code": locality_details["pin_code"],
        "society_type": society_type,
        "distance_to_metro_km": round(distance_to_metro, 2),
        "distance_to_it_park_km": round(distance_to_it_park, 2),
        "distance_to_school_km": round(distance_to_school, 2),
        "distance_to_hospital_km": round(distance_to_hospital, 2),
        "rental_price": round(rental_price, 2),
        "purchase_price": round(purchase_price, 2),
        "price_per_sqft": round(purchase_price / max(carpet_area, 1), 2),
        "brokerage_type": brokerage_type,
        "days_since_listed": days_since_listed,
    }


def inject_data_quality_noise(dataframe: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject limited missing values and duplicates to simulate real-world listings."""

    df = dataframe.copy()
    missing_columns = [
        "super_built_up_area_sqft",
        "maintenance_charges",
        "flooring_type",
        "parking_type",
        "water_supply",
        "brokerage_type",
    ]
    for column in missing_columns:
        missing_indices = rng.choice(df.index, size=max(10, int(0.03 * len(df))), replace=False)
        df.loc[missing_indices, column] = np.nan

    duplicate_count = max(20, int(len(df) * 0.04))
    duplicates = df.sample(duplicate_count, random_state=RANDOM_STATE).copy()
    duplicates["listing_id"] = [f"{listing_id}-DUP" for listing_id in duplicates["listing_id"]]
    duplicates["days_since_listed"] = duplicates["days_since_listed"].astype(int) + rng.integers(0, 3, len(duplicates))
    duplicates["rental_price"] = duplicates["rental_price"] * rng.uniform(0.99, 1.02, len(duplicates))
    duplicates["purchase_price"] = duplicates["purchase_price"] * rng.uniform(0.99, 1.02, len(duplicates))
    return pd.concat([df, duplicates], ignore_index=True)


def generate_sample_dataset(
    record_count: int = 2200,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate and save a realistic Pune real estate sample dataset."""

    ensure_directories()
    logger = get_logger("sample_data_generator", RAW_DATA_DIR / "sample_generation.log")
    rng = np.random.default_rng(RANDOM_STATE)

    localities = list(LOCALITY_TIER_MAP.keys())
    locality_weights = np.array([0.07, 0.06, 0.08, 0.08, 0.08, 0.07, 0.09, 0.05, 0.08, 0.04, 0.06, 0.09, 0.08, 0.04, 0.03])
    locality_weights = locality_weights / locality_weights.sum()

    rows = []
    for index in range(record_count):
        locality = str(rng.choice(localities, p=locality_weights))
        rows.append(create_listing(locality, index=index + 1, rng=rng))

    dataframe = pd.DataFrame(rows, columns=LISTING_SCHEMA)
    dataframe = inject_data_quality_noise(dataframe, rng)

    output_file = output_path or RAW_DATA_DIR / "pune_property_listings_sample.csv"
    dataframe.to_csv(output_file, index=False)

    logger.info("Sample dataset generated with %s rows at %s", len(dataframe), output_file)
    logger.info("Rental median: %.2f INR/month", dataframe["rental_price"].median())
    logger.info("Purchase median: %.2f INR", dataframe["purchase_price"].median())
    return dataframe


def main() -> None:
    """CLI entry point for dataset generation."""

    dataframe = generate_sample_dataset()
    print(f"Saved {len(dataframe)} sample listings to {RAW_DATA_DIR / 'pune_property_listings_sample.csv'}")


if __name__ == "__main__":
    main()
