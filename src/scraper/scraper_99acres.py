"""99acres scraper scaffold for Pune rental and sale listings."""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.common import LISTING_SCHEMA, LOCALITY_TIER_MAP, RAW_DATA_DIR, ensure_directories, get_logger

BASE_URL = "https://www.99acres.com/search/property"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class SearchTask:
    """Represent a scrape search for a specific locality and listing intent."""

    locality: str
    listing_intent: str


def build_search_url(task: SearchTask, page: int) -> str:
    """Build a 99acres search URL for Pune listings."""

    query = {
        "city": "19",
        "keyword": f"{task.locality}, Pune",
        "page": page,
        "preference": "S" if task.listing_intent == "sale" else "R",
    }
    return f"{BASE_URL}?{urlencode(query)}"


def safe_text(element, default: str = "") -> str:
    """Return stripped text from a BeautifulSoup element."""

    return element.get_text(strip=True) if element else default


def extract_numeric(text: str) -> float | None:
    """Extract the first numeric value from a text snippet."""

    allowed = "".join(character if character.isdigit() or character == "." else " " for character in text)
    parts = [part for part in allowed.split() if part]
    return float(parts[0]) if parts else None


def parse_card(card, locality: str, listing_intent: str) -> dict[str, object]:
    """Parse a listing card using resilient fallback selectors."""

    price_text = safe_text(card.select_one("[data-label='PRICE']")) or safe_text(card.select_one(".srpTuple__price"))
    title_text = safe_text(card.select_one("a"))
    area_text = safe_text(card.select_one("[data-label='BUILTUP_AREA']")) or safe_text(card.select_one(".srpTuple__area"))
    bhk_text = safe_text(card.select_one("[data-label='PROPERTY_TYPE']")) or title_text
    description = safe_text(card.select_one(".srpTuple__propertyDescription")) or safe_text(card.select_one(".srpTuple__tupleDetails"))

    rental_price = extract_numeric(price_text) if listing_intent == "rent" else None
    purchase_price = extract_numeric(price_text) if listing_intent == "sale" else None
    carpet_area = extract_numeric(area_text)

    row = {column: None for column in LISTING_SCHEMA}
    row.update(
        {
            "listing_id": card.get("id", "") if hasattr(card, "get") else "",
            "source": "99acres",
            "listing_url": card.select_one("a")["href"] if card.select_one("a") and card.select_one("a").has_attr("href") else "",
            "bhk_type": bhk_text or "Not Specified",
            "carpet_area_sqft": carpet_area,
            "super_built_up_area_sqft": carpet_area,
            "furnishing_status": "Not Specified",
            "flooring_type": "Not Specified",
            "parking_type": "Not Specified",
            "water_supply": "Not Specified",
            "locality": locality,
            "pin_code": "",
            "society_type": "Not Specified",
            "rental_price": rental_price,
            "purchase_price": purchase_price,
            "price_per_sqft": (purchase_price / carpet_area) if purchase_price and carpet_area else None,
            "brokerage_type": "Not Specified",
            "days_since_listed": None,
        }
    )

    if description:
        lowered = description.lower()
        row["lift_available"] = int("lift" in lowered)
        row["gym"] = int("gym" in lowered)
        row["swimming_pool"] = int("pool" in lowered)
        row["clubhouse"] = int("clubhouse" in lowered)
        row["security"] = int("security" in lowered)
        row["gated_community"] = int("gated" in lowered)
        row["power_backup"] = int("backup" in lowered)
    return row


def parse_listing_cards(html: str, locality: str, listing_intent: str) -> list[dict[str, object]]:
    """Parse all relevant listing cards from a search result page."""

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(".srpTuple") or soup.select("[data-testid='tuple']")
    return [parse_card(card, locality=locality, listing_intent=listing_intent) for card in cards]


def scrape_99acres(max_pages: int = 8, request_timeout: int = 30) -> pd.DataFrame:
    """Scrape Pune listings from 99acres with polite delays and error logging."""

    ensure_directories()
    logger = get_logger("scraper_99acres", RAW_DATA_DIR / "99acres_scrape.log")
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows: list[dict[str, object]] = []
    tasks: Iterable[SearchTask] = [
        SearchTask(locality=locality, listing_intent=listing_intent)
        for locality in LOCALITY_TIER_MAP
        for listing_intent in ("rent", "sale")
    ]

    for task in tasks:
        logger.info("Scraping %s listings for %s", task.listing_intent, task.locality)
        for page in range(1, max_pages + 1):
            url = build_search_url(task, page=page)
            try:
                response = session.get(url, timeout=request_timeout)
                response.raise_for_status()
                rows = parse_listing_cards(response.text, locality=task.locality, listing_intent=task.listing_intent)
                if not rows:
                    logger.warning("No cards found for %s page %s. The site may be blocking requests.", task.locality, page)
                    break
                all_rows.extend(rows)
                logger.info("Captured %s rows from %s page %s", len(rows), task.locality, page)
            except requests.RequestException as error:
                logger.error("Request failed for %s page %s: %s", task.locality, page, error)
                break
            except Exception as error:  # pragma: no cover - defensive parser fallback
                logger.exception("Unexpected parsing error for %s page %s: %s", task.locality, page, error)
                break
            time.sleep(random.uniform(3, 8))

    dataframe = pd.DataFrame(all_rows, columns=LISTING_SCHEMA)
    output_path = RAW_DATA_DIR / "99acres_raw.csv"
    dataframe.to_csv(output_path, index=False)
    logger.info("Saved %s scraped rows to %s", len(dataframe), output_path)
    return dataframe


def main() -> None:
    """CLI entry point for the scraper."""

    dataframe = scrape_99acres()
    print(f"Scraped {len(dataframe)} rows into {Path(RAW_DATA_DIR / '99acres_raw.csv')}")


if __name__ == "__main__":
    main()
