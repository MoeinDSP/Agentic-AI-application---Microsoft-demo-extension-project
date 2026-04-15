from __future__ import annotations

import uuid

import httpx

from app.core.config import settings
from app.models import Location, RestaurantCandidate

_PLACES_BASE = "https://maps.googleapis.com/maps/api/place"

# Budget → Places API price_level (0-4)
_BUDGET_TO_PRICE_LEVEL = [
    (15.0, 1),
    (35.0, 2),
    (70.0, 3),
]

# Types to exclude from cuisine display
_SKIP_TYPES = {
    "food", "point_of_interest", "establishment",
    "restaurant", "store", "health", "premise",
}

# Map Places API type strings to human-readable cuisine labels
_CUISINE_MAP: dict[str, str] = {
    "italian_restaurant":       "Italian",
    "pizza_restaurant":         "Pizza",
    "cafe":                     "Café",
    "bakery":                   "Bakery",
    "bar":                      "Bar",
    "fast_food_restaurant":     "Fast Food",
    "meal_takeaway":            "Takeaway",
    "meal_delivery":            "Delivery",
    "japanese_restaurant":      "Japanese",
    "chinese_restaurant":       "Chinese",
    "american_restaurant":      "American",
    "french_restaurant":        "French",
    "mediterranean_restaurant": "Mediterranean",
    "seafood_restaurant":       "Seafood",
    "steak_house":              "Steakhouse",
    "vegetarian_restaurant":    "Vegetarian",
    "vegan_restaurant":         "Vegan",
    "sushi_restaurant":         "Sushi",
    "indian_restaurant":        "Indian",
    "mexican_restaurant":       "Mexican",
    "greek_restaurant":         "Greek",
    "middle_eastern_restaurant":"Middle Eastern",
    "brunch_restaurant":        "Brunch",
    "breakfast_restaurant":     "Breakfast",
    "wine_bar":                 "Wine Bar",
}


def _budget_to_max_price(budget: float) -> int | None:
    for threshold, level in _BUDGET_TO_PRICE_LEVEL:
        if budget <= threshold:
            return level
    return None


def _parse_cuisines(types: list[str]) -> list[str]:
    result = []
    for t in types:
        if t in _SKIP_TYPES:
            continue
        label = _CUISINE_MAP.get(t)
        if label:
            result.append(label)
        else:
            # Clean up unknown types: "some_type" → "Some Type"
            cleaned = t.replace("_", " ").title()
            result.append(cleaned)
    return result[:3]


async def search_restaurants(
    latitude: float,
    longitude: float,
    radius_meters: int = 1000,
    meal_slot: str = "lunch",
    budget_per_person: float | None = None,
    preferences: list[str] | None = None,
) -> list[dict]:
    """
    Search nearby restaurants using the Google Places Nearby Search API (Legacy).

    Returns:
        List of restaurant dicts compatible with RestaurantCandidate.
    """
    slot_defaults = {
        "breakfast": "breakfast cafe",
        "lunch":     "lunch restaurant",
        "dinner":    "dinner restaurant",
    }

    keyword = (
        " ".join(preferences)
        if preferences
        else slot_defaults.get(meal_slot, "restaurant")
    )

    params: dict = {
        "location": f"{latitude},{longitude}",
        "radius":   radius_meters,
        "type":     "restaurant",
        "keyword":  keyword,
        "key":      settings.google_places_api_key,
    }

    if budget_per_person is not None:
        max_price = _budget_to_max_price(budget_per_person)
        if max_price is not None:
            params["maxprice"] = max_price

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{_PLACES_BASE}/nearbysearch/json",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    candidates: list[dict] = []

    for place in data.get("results", [])[:6]:
        geo      = place.get("geometry", {}).get("location", {})
        types    = place.get("types", [])
        cuisines = _parse_cuisines(types)

        # price_level: 0-4 integer (may be absent for some venues)
        raw_price = place.get("price_level")
        price_level = int(raw_price) if raw_price is not None else None

        candidates.append(
            RestaurantCandidate(
                id=place.get("place_id", str(uuid.uuid4())),
                name=place.get("name", "Unknown"),
                location=Location(
                    latitude=geo.get("lat", 0.0),
                    longitude=geo.get("lng", 0.0),
                    address=place.get("vicinity"),
                ),
                price_level=price_level,
                cuisines=cuisines or None,
                rating=place.get("rating"),
                summary=place.get("editorial_summary", {}).get("overview"),
            ).model_dump()
        )

    return candidates