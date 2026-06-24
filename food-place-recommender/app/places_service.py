from __future__ import annotations

import uuid

import httpx

from environment_service import env
from schemas import Location, RestaurantCandidate

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
    "italian_restaurant":        "Italian",
    "pizza_restaurant":          "Pizza",
    "cafe":                      "Café",
    "bakery":                    "Bakery",
    "bar":                       "Bar",
    "fast_food_restaurant":      "Fast Food",
    "meal_takeaway":             "Takeaway",
    "meal_delivery":             "Delivery",
    "japanese_restaurant":       "Japanese",
    "chinese_restaurant":        "Chinese",
    "american_restaurant":       "American",
    "french_restaurant":         "French",
    "mediterranean_restaurant":  "Mediterranean",
    "seafood_restaurant":        "Seafood",
    "steak_house":               "Steakhouse",
    "vegetarian_restaurant":     "Vegetarian",
    "vegan_restaurant":          "Vegan",
    "sushi_restaurant":          "Sushi",
    "indian_restaurant":         "Indian",
    "mexican_restaurant":        "Mexican",
    "greek_restaurant":          "Greek",
    "middle_eastern_restaurant": "Middle Eastern",
    "brunch_restaurant":         "Brunch",
    "breakfast_restaurant":      "Breakfast",
    "wine_bar":                  "Wine Bar",
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
            result.append(t.replace("_", " ").title())
    return result[:3]


class PlacesApiError(Exception):
    """Raised when the Google Places API returns a non-OK status."""


# Statuses that mean "no results" — not an error, just an empty search.
_EMPTY_STATUSES = {"ZERO_RESULTS", "OK"}


class PlacesService:
    """Wraps the Google Places Nearby Search API (Legacy) for restaurant discovery."""

    @staticmethod
    async def search_restaurants(
        latitude: float,
        longitude: float,
        radius_meters: int = 1000,
        meal_slot: str = "lunch",
        budget_per_person: float | None = None,
        preferences: list[str] | None = None,
    ) -> list[RestaurantCandidate]:
        """
        Search nearby restaurants using the Google Places Nearby Search API (Legacy).

        Returns up to 6 RestaurantCandidate objects for the given location and parameters.
        Raises PlacesApiError when the API signals a configuration or quota problem.
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
            "key":      env.google_places_api_key,
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

        status = data.get("status", "UNKNOWN")
        print(f"[PlacesService] status={status} | results={len(data.get('results', []))}")

        if status not in _EMPTY_STATUSES:
            error_message = data.get("error_message", "no details provided")
            raise PlacesApiError(
                f"Google Places API returned status '{status}': {error_message}"
            )

        candidates: list[RestaurantCandidate] = []

        for place in data.get("results", [])[:6]:
            geo = place.get("geometry", {}).get("location", {})
            cuisines = _parse_cuisines(place.get("types", []))

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
                )
            )

        return candidates
