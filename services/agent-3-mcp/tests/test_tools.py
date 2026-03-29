from agent3_mcp.models.tools import Coordinates, RouteEstimateRequest
from agent3_mcp.services.tools import ToolService


def test_route_estimate_varies_by_mode() -> None:
    service = ToolService()
    origin = Coordinates(lat=41.9028, lng=12.4964)
    destination = Coordinates(lat=41.8902, lng=12.4922)

    walk = service.estimate_route(
        RouteEstimateRequest(origin=origin, destination=destination, mode="walk")
    )
    transit = service.estimate_route(
        RouteEstimateRequest(origin=origin, destination=destination, mode="transit")
    )
    drive = service.estimate_route(
        RouteEstimateRequest(origin=origin, destination=destination, mode="drive")
    )

    assert walk.mode == "walk"
    assert transit.mode == "transit"
    assert drive.mode == "drive"
    assert walk.estimated_duration_minutes > transit.estimated_duration_minutes
    assert transit.estimated_duration_minutes > drive.estimated_duration_minutes
    assert "transport_mode=walk" in walk.notes
