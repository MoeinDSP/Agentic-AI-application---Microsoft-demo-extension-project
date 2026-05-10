from datetime import datetime
from functools import lru_cache

import httpx

from agent3.core.config import Settings, get_settings
from agent3.core.logging import get_logger
from agent3.models.mcp import MCPRouteEstimateRequest, MCPRouteEstimateResponse, TravelEstimate
from agent3.models.plan import DEFAULT_TRANSPORT_MODE, SUPPORTED_TRANSPORT_MODES, Coordinates


class RouteEstimationError(Exception):
    pass


class MCPRouteClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("agent3.mcp_client", environment=settings.environment)

    def estimate_route(
        self,
        *,
        origin: Coordinates,
        destination: Coordinates,
        transport_preferences: list[str],
        departure_time: datetime | None = None,
    ) -> TravelEstimate:
        mode = self.resolve_transport_mode(transport_preferences)
        request = MCPRouteEstimateRequest(
            origin=origin,
            destination=destination,
            mode=mode,
            departure_time=departure_time,
        )

        try:
            with httpx.Client(timeout=self._settings.mcp_timeout_seconds) as client:
                response = client.post(
                    f"{self._settings.mcp_base_url.rstrip('/')}/v1/tools/route-estimate",
                    json=request.model_dump(mode="json"),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            self._logger.warning(
                "route_estimation_failed",
                extra={
                    "event": "mcp_route_estimation_failed",
                    "mcp_base_url": self._settings.mcp_base_url,
                    "mode": mode,
                },
            )
            raise RouteEstimationError("Route estimation request failed") from exc

        try:
            payload = MCPRouteEstimateResponse.model_validate(response.json())
        except ValueError as exc:
            self._logger.warning(
                "route_estimation_invalid_response",
                extra={
                    "event": "mcp_route_estimation_invalid_response",
                    "mcp_base_url": self._settings.mcp_base_url,
                    "mode": mode,
                },
            )
            raise RouteEstimationError("Route estimation response was invalid") from exc

        return TravelEstimate(
            source="mcp",
            mode=payload.mode,
            estimated_duration_minutes=payload.estimated_duration_minutes,
            notes=payload.notes,
        )

    def resolve_transport_mode(self, transport_preferences: list[str]) -> str:
        if not transport_preferences:
            return self._settings.default_transport_mode or DEFAULT_TRANSPORT_MODE

        mode = transport_preferences[0]
        if mode not in SUPPORTED_TRANSPORT_MODES:
            raise RouteEstimationError("Transport mode is unsupported")
        return mode


@lru_cache(maxsize=1)
def get_mcp_route_client() -> MCPRouteClient:
    return MCPRouteClient(get_settings())
