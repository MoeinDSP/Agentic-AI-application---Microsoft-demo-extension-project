from __future__ import annotations

import uuid

import httpx

from app.core.config import settings
from app.models import Location, RestaurantCandidate

_PLACES_BASE = "https://maps.googleapis.com/maps/api/place"

# Google Places price_level thresholds (EUR)
_BUDGET_TO_PRICE_LEVEL = [
    (15.0, 1),
    (35.0, 2),
    (70.0, 3),
]

# Types returned by Places API that carry no cuisine meaning
_SKIP_TYPES = {"food", "point_of_interest", "establishment", "restaurant"}


def _budget_to_max_price(budget: float) -> int | None:
    """Map a per-person EUR budget to a Google Places maxprice level (1-4)."""
    for threshold, level in _BUDGET_TO_PRICE_LEVEL:
        if budget < threshold:
            return level
    return None  # no restriction above €70


async def search_restaurants(
    latitude: float,
    longitude: float,
    radius_meters: int = 1000,
    meal_slot: str = "lunch",
    budget_per_person: float | None = None,
    preferences: list[str] | None = None,
) -> list[dict]:
    """
    Search nearby restaurants using the Google Places Nearby Search API.

    Args:
        latitude:          Latitude of the search centre.
        longitude:         Longitude of the search centre.
        radius_meters:     Search radius in metres (100 – 50 000).
        meal_slot:         'breakfast', 'lunch', or 'dinner'.
        budget_per_person: Optional max spend per person in EUR.
        preferences:       Optional cuisine / dietary keywords.

    Returns:
        List of restaurant dicts compatible with RestaurantCandidate.
    """
    # Build search keyword
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
        geo = place.get("geometry", {}).get("location", {})

        cuisines = [
            t for t in place.get("types", [])
            if t not in _SKIP_TYPES
        ][:3]

        candidates.append(
            RestaurantCandidate(
                id=place.get("place_id", str(uuid.uuid4())),
                name=place.get("name", "Unknown"),
                location=Location(
                    latitude=geo.get("lat", 0.0),
                    longitude=geo.get("lng", 0.0),
                    address=place.get("vicinity"),
                ),
                price_level=place.get("price_level"),
                cuisines=cuisines or None,
                rating=place.get("rating"),
                summary=place.get("editorial_summary", {}).get("overview"),
            ).model_dump()
        )

    return candidates