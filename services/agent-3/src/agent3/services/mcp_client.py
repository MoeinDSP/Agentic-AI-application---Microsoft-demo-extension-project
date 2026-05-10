from datetime import datetime
from functools import lru_cache
from time import time

import httpx
from google.auth.transport.requests import Request
from google.oauth2 import id_token

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
        self._google_request = Request()
        self._cached_id_token: str | None = None
        self._cached_id_token_audience: str | None = None
        self._cached_id_token_expires_at: float = 0.0

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
                    headers=self._build_headers(),
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

    def _build_headers(self) -> dict[str, str]:
        if self._settings.mcp_auth_mode == "none":
            return {}
        if self._settings.mcp_auth_mode == "gcp_id_token":
            return {"Authorization": f"Bearer {self._get_mcp_id_token()}"}
        raise RouteEstimationError("MCP auth mode is unsupported")

    def _get_mcp_id_token(self) -> str:
        audience = self._settings.mcp_base_url.rstrip("/")
        now = time()
        if (
            self._cached_id_token
            and self._cached_id_token_audience == audience
            and now < self._cached_id_token_expires_at
        ):
            return self._cached_id_token

        try:
            token = id_token.fetch_id_token(self._google_request, audience)
        except Exception as exc:  # pragma: no cover - exact google auth exceptions vary
            self._logger.warning(
                "mcp_id_token_fetch_failed",
                extra={
                    "event": "mcp_id_token_fetch_failed",
                    "mcp_base_url": self._settings.mcp_base_url,
                },
            )
            raise RouteEstimationError("Failed to fetch MCP identity token") from exc

        self._cached_id_token = token
        self._cached_id_token_audience = audience
        self._cached_id_token_expires_at = now + 50 * 60
        return token


@lru_cache(maxsize=1)
def get_mcp_route_client() -> MCPRouteClient:
    return MCPRouteClient(get_settings())
