from functools import lru_cache

from agent3_mcp.models.tools import (
    PlaceDetail,
    PlaceDetailsRequest,
    PlaceDetailsResponse,
    RouteEstimateRequest,
    RouteEstimateResponse,
)


class ToolService:
    def estimate_route(self, request: RouteEstimateRequest) -> RouteEstimateResponse:
        lat_gap = abs(request.origin.lat - request.destination.lat)
        lng_gap = abs(request.origin.lng - request.destination.lng)
        estimated_distance_km = round((lat_gap + lng_gap) * 55, 2)
        mode_multiplier = {
            "walk": 3.0,
            "drive": 1.2,
            "transit": 1.8,
        }[request.mode]
        minimum_duration = {
            "walk": 10,
            "drive": 5,
            "transit": 8,
        }[request.mode]
        estimated_duration_minutes = max(
            minimum_duration,
            int(estimated_distance_km * mode_multiplier),
        )

        return RouteEstimateResponse(
            mode=request.mode,
            estimated_distance_km=estimated_distance_km,
            estimated_duration_minutes=estimated_duration_minutes,
            notes=[
                "placeholder_route_estimate",
                f"transport_mode={request.mode}",
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


@lru_cache(maxsize=1)
def get_tool_service() -> ToolService:
    return ToolService()
