from __future__ import annotations

from typing import Optional

import httpx

from environment_service import env
from schemas import PlaceDetail, PlaceDetails, RouteEstimate

GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = "routes.duration,routes.distanceMeters,routes.warnings"
GOOGLE_TRAVEL_MODES = {
    "walking": "WALK",
    "driving": "DRIVE",
    "transit": "TRANSIT",
    "bicycling": "BICYCLE",
}


class RouteEstimationError(Exception):
    """Raised when a route estimate cannot be produced."""


class RoutesService:
    """Wraps the Google Routes API (computeRoutes) for travel-time estimation."""

    @staticmethod
    def estimate_route(
        origin_latitude: float,
        origin_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
        mode: str,
        departure_time: Optional[str] = None,
    ) -> RouteEstimate:
        """Estimate distance and duration between two coordinates for a transport mode."""
        normalized_mode = mode.strip().lower()
        if normalized_mode not in GOOGLE_TRAVEL_MODES:
            raise RouteEstimationError(
                "mode must be one of: walking, driving, transit, bicycling"
            )
        if not env.google_maps_api_key:
            raise RouteEstimationError(
                "GOOGLE_MAPS_API_KEY must be set for route estimation"
            )

        payload: dict = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": origin_latitude,
                        "longitude": origin_longitude,
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": destination_latitude,
                        "longitude": destination_longitude,
                    }
                }
            },
            "travelMode": GOOGLE_TRAVEL_MODES[normalized_mode],
            "units": "METRIC",
        }
        if departure_time:
            payload["departureTime"] = departure_time

        with httpx.Client(timeout=env.google_routes_timeout_seconds) as client:
            response = client.post(
                GOOGLE_ROUTES_URL,
                json=payload,
                headers={
                    "X-Goog-Api-Key": env.google_maps_api_key,
                    "X-Goog-FieldMask": FIELD_MASK,
                },
            )
            response.raise_for_status()

        routes = response.json().get("routes", [])
        if not routes:
            raise RouteEstimationError("Google Routes response did not include any routes")
        route = routes[0]

        return RouteEstimate(
            mode=normalized_mode,
            estimated_distance_km=round(route["distanceMeters"] / 1000, 2),
            estimated_duration_minutes=RoutesService._duration_to_minutes(route["duration"]),
            notes=[
                "provider=google_routes",
                f"transport_mode={normalized_mode}",
                *route.get("warnings", []),
            ],
        )

    @staticmethod
    def get_place_details(place_ids: list[str]) -> PlaceDetails:
        """Return deterministic placeholder metadata for the given place ids."""
        places = [
            PlaceDetail(
                place_id=place_id,
                display_name=place_id.replace("-", " ").title(),
                category="placeholder",
                summary="Deterministic placeholder place details.",
            )
            for place_id in sorted(place_ids)
        ]
        return PlaceDetails(places=places, notes=["placeholder_place_details"])

    @staticmethod
    def _duration_to_minutes(value: str) -> int:
        if not value.endswith("s"):
            raise RouteEstimationError("Google Routes duration must use seconds format")
        seconds = float(value[:-1])
        return max(1, int(round(seconds / 60)))
