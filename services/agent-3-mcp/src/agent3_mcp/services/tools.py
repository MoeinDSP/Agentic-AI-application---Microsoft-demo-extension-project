from datetime import UTC, datetime
from functools import lru_cache

import httpx

from agent3_mcp.core.config import Settings, get_settings
from agent3_mcp.models.tools import (
    PlaceDetail,
    PlaceDetailsRequest,
    PlaceDetailsResponse,
    RouteEstimateRequest,
    RouteEstimateResponse,
)

GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = "routes.duration,routes.distanceMeters,routes.warnings"
GOOGLE_TRAVEL_MODES = {
    "walking": "WALK",
    "driving": "DRIVE",
    "transit": "TRANSIT",
    "bicycling": "BICYCLE",
}


class ToolService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def estimate_route(self, request: RouteEstimateRequest) -> RouteEstimateResponse:
        if not self._settings.google_maps_api_key:
            raise ValueError(
                "AGENT3_MCP_GOOGLE_MAPS_API_KEY must be set for route estimation"
            )

        payload = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": request.origin.lat,
                        "longitude": request.origin.lng,
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": request.destination.lat,
                        "longitude": request.destination.lng,
                    }
                }
            },
            "travelMode": GOOGLE_TRAVEL_MODES[request.mode],
            "units": "METRIC",
        }
        if request.mode == "transit":
            payload["departureTime"] = datetime.now(UTC).isoformat()

        with httpx.Client(timeout=self._settings.google_routes_timeout_seconds) as client:
            response = client.post(
                GOOGLE_ROUTES_URL,
                json=payload,
                headers={
                    "X-Goog-Api-Key": self._settings.google_maps_api_key,
                    "X-Goog-FieldMask": FIELD_MASK,
                },
            )
            response.raise_for_status()

        routes = response.json().get("routes", [])
        if not routes:
            raise ValueError("Google Routes response did not include any routes")
        route = routes[0]

        return RouteEstimateResponse(
            mode=request.mode,
            estimated_distance_km=round(route["distanceMeters"] / 1000, 2),
            estimated_duration_minutes=self._duration_to_minutes(route["duration"]),
            notes=[
                "provider=google_routes",
                f"transport_mode={request.mode}",
                *route.get("warnings", []),
            ],
        )

    def get_place_details(self, request: PlaceDetailsRequest) -> PlaceDetailsResponse:
        places = [
            PlaceDetail(
                place_id=place_id,
                display_name=place_id.replace("-", " ").title(),
                category="placeholder",
                summary="Deterministic placeholder place details.",
            )
            for place_id in sorted(request.place_ids)
        ]
        return PlaceDetailsResponse(
            places=places,
            notes=["placeholder_place_details"],
        )

    def _duration_to_minutes(self, value: str) -> int:
        if not value.endswith("s"):
            raise ValueError("Google Routes duration must use seconds format")
        seconds = float(value[:-1])
        return max(1, int(round(seconds / 60)))


@lru_cache(maxsize=1)
def get_tool_service() -> ToolService:
    return ToolService(get_settings())
