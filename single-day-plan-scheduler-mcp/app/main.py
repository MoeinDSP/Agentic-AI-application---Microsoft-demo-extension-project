from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP
from routes_service import RoutesService
from schemas import PlaceDetails, RouteEstimate
from starlette.types import ASGIApp, Receive, Scope, Send

mcp = FastMCP(
    name="single-day-plan-scheduler-mcp",
    instructions=(
        "Routing and place tools for the single-day-plan-scheduler agent, backed by the "
        "Google Maps Platform Routes API."
    ),
    host="0.0.0.0",
    stateless_http=True,
)


@mcp.tool()
def route_estimate(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    mode: str,
    departure_time: Optional[str] = None,
) -> RouteEstimate:
    """Estimate travel distance and duration between two coordinates.

    Args:
        origin_latitude: Latitude of the origin (WGS84 degrees).
        origin_longitude: Longitude of the origin (WGS84 degrees).
        destination_latitude: Latitude of the destination (WGS84 degrees).
        destination_longitude: Longitude of the destination (WGS84 degrees).
        mode: One of 'walking', 'driving', 'transit', 'bicycling'.
        departure_time: Optional ISO 8601 departure time.

    Returns:
        A RouteEstimate with the mode, estimated distance in km, estimated duration in
        minutes, and provider notes.
    """
    return RoutesService.estimate_route(
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
        destination_latitude=destination_latitude,
        destination_longitude=destination_longitude,
        mode=mode,
        departure_time=departure_time,
    )


@mcp.tool()
def place_details(place_ids: list[str]) -> PlaceDetails:
    """Return metadata for a list of place ids.

    Args:
        place_ids: The place ids to look up.

    Returns:
        A PlaceDetails object with one entry per requested place id.
    """
    return RoutesService.get_place_details(place_ids)


def _add_health_route(asgi_app: ASGIApp) -> ASGIApp:
    """Wrap an ASGI app with a /health endpoint that always returns 200 ok."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/health":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
            return
        await asgi_app(scope, receive, send)

    return app


app = _add_health_route(mcp.streamable_http_app())
