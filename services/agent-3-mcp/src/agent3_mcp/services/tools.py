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
        estimated_duration_minutes = max(10, int(estimated_distance_km * 3))

        return RouteEstimateResponse(
            mode=request.mode,
            estimated_distance_km=estimated_distance_km,
            estimated_duration_minutes=estimated_duration_minutes,
            notes=["placeholder_route_estimate"],
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
