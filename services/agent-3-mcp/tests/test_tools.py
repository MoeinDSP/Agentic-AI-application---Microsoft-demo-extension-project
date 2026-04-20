from agent3_mcp.models.tools import Coordinates, RouteEstimateRequest
from agent3_mcp.services.tools import ToolService


def test_route_estimate_varies_by_mode() -> None:
    service = ToolService()
    origin = Coordinates(lat=41.9028, lng=12.4964)
    destination = Coordinates(lat=41.8902, lng=12.4922)

    walking = service.estimate_route(
        RouteEstimateRequest(origin=origin, destination=destination, mode="walking")
    )
    transit = service.estimate_route(
        RouteEstimateRequest(origin=origin, destination=destination, mode="transit")
    )
    driving = service.estimate_route(
        RouteEstimateRequest(origin=origin, destination=destination, mode="driving")
    )
    bicycling = service.estimate_route(
        RouteEstimateRequest(origin=origin, destination=destination, mode="bicycling")
    )

    assert walking.mode == "walking"
    assert transit.mode == "transit"
    assert driving.mode == "driving"
    assert bicycling.mode == "bicycling"
    assert walking.estimated_duration_minutes > bicycling.estimated_duration_minutes
    assert transit.estimated_duration_minutes > bicycling.estimated_duration_minutes
    assert transit.estimated_duration_minutes > driving.estimated_duration_minutes
    assert bicycling.estimated_duration_minutes > driving.estimated_duration_minutes
    assert "transport_mode=walking" in walking.notes
