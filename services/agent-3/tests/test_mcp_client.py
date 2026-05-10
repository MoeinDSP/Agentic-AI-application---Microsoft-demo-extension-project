from datetime import datetime

import httpx
import pytest

from agent3.core.config import Settings
from agent3.models.plan import Coordinates
from agent3.services.mcp_client import MCPRouteClient, RouteEstimationError


class _MockResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, object]:
        return self._payload


class _MockClient:
    def __init__(
        self,
        response: _MockResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error

    def __enter__(self) -> "_MockClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(
        self,
        url: str,
        json: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> _MockResponse:
        if self._error is not None:
            raise self._error
        assert url.endswith("/v1/tools/route-estimate")
        return self._response or _MockResponse({})


def test_mcp_route_client_maps_success_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout: _MockClient(
            response=_MockResponse(
                {
                    "mode": "walking",
                    "estimated_distance_km": 1.2,
                    "estimated_duration_minutes": 14,
                    "notes": ["mock_success"],
                }
            )
        ),
    )
    client = MCPRouteClient(Settings())

    response = client.estimate_route(
        origin=Coordinates(lat=41.9, lng=12.4),
        destination=Coordinates(lat=41.8, lng=12.5),
        transport_preferences=["walking"],
    )

    assert response.source == "mcp"
    assert response.estimated_duration_minutes == 14
    assert response.notes == ["mock_success"]


def test_mcp_route_client_uses_default_transport_mode_when_preferences_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturingClient(_MockClient):
        def post(
            self,
            url: str,
            json: dict[str, object],
            headers: dict[str, str] | None = None,
        ) -> _MockResponse:
            captured.update(json)
            return _MockResponse(
                {
                    "mode": json["mode"],
                    "estimated_distance_km": 1.2,
                    "estimated_duration_minutes": 14,
                    "notes": ["mock_success"],
                }
            )

    monkeypatch.setattr(httpx, "Client", lambda timeout: _CapturingClient())
    client = MCPRouteClient(Settings())

    response = client.estimate_route(
        origin=Coordinates(lat=41.9, lng=12.4),
        destination=Coordinates(lat=41.8, lng=12.5),
        transport_preferences=[],
    )

    assert captured["mode"] == "walking"
    assert response.mode == "walking"


def test_mcp_route_client_uses_explicit_transport_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturingClient(_MockClient):
        def post(
            self,
            url: str,
            json: dict[str, object],
            headers: dict[str, str] | None = None,
        ) -> _MockResponse:
            captured.update(json)
            return _MockResponse(
                {
                    "mode": json["mode"],
                    "estimated_distance_km": 1.2,
                    "estimated_duration_minutes": 8,
                    "notes": ["mock_success"],
                }
            )

    monkeypatch.setattr(httpx, "Client", lambda timeout: _CapturingClient())
    client = MCPRouteClient(Settings())

    response = client.estimate_route(
        origin=Coordinates(lat=41.9, lng=12.4),
        destination=Coordinates(lat=41.8, lng=12.5),
        transport_preferences=["driving"],
    )

    assert captured["mode"] == "driving"
    assert response.mode == "driving"


def test_mcp_route_client_serializes_departure_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturingClient(_MockClient):
        def post(
            self,
            url: str,
            json: dict[str, object],
            headers: dict[str, str] | None = None,
        ) -> _MockResponse:
            captured.update(json)
            return _MockResponse(
                {
                    "mode": json["mode"],
                    "estimated_distance_km": 3.1,
                    "estimated_duration_minutes": 18,
                    "notes": ["mock_success"],
                }
            )

    monkeypatch.setattr(httpx, "Client", lambda timeout: _CapturingClient())
    client = MCPRouteClient(Settings())
    departure_time = datetime.fromisoformat("2026-05-02T08:15:00")

    client.estimate_route(
        origin=Coordinates(lat=41.9, lng=12.4),
        destination=Coordinates(lat=41.8, lng=12.5),
        transport_preferences=["transit"],
        departure_time=departure_time,
    )

    assert captured["departure_time"] == "2026-05-02T08:15:00"


def test_mcp_route_client_sends_no_auth_header_in_none_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_headers: dict[str, str] | None = None

    class _CapturingClient(_MockClient):
        def post(
            self,
            url: str,
            json: dict[str, object],
            headers: dict[str, str] | None = None,
        ) -> _MockResponse:
            nonlocal captured_headers
            captured_headers = headers
            return _MockResponse(
                {
                    "mode": json["mode"],
                    "estimated_distance_km": 1.2,
                    "estimated_duration_minutes": 14,
                    "notes": ["mock_success"],
                }
            )

    monkeypatch.setattr(httpx, "Client", lambda timeout: _CapturingClient())
    client = MCPRouteClient(Settings(mcp_auth_mode="none"))

    client.estimate_route(
        origin=Coordinates(lat=41.9, lng=12.4),
        destination=Coordinates(lat=41.8, lng=12.5),
        transport_preferences=["walking"],
    )

    assert captured_headers == {}


def test_mcp_route_client_sends_bearer_token_in_gcp_id_token_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_headers: dict[str, str] | None = None

    class _CapturingClient(_MockClient):
        def post(
            self,
            url: str,
            json: dict[str, object],
            headers: dict[str, str] | None = None,
        ) -> _MockResponse:
            nonlocal captured_headers
            captured_headers = headers
            return _MockResponse(
                {
                    "mode": json["mode"],
                    "estimated_distance_km": 1.2,
                    "estimated_duration_minutes": 14,
                    "notes": ["mock_success"],
                }
            )

    monkeypatch.setattr(httpx, "Client", lambda timeout: _CapturingClient())
    monkeypatch.setattr(
        "agent3.services.mcp_client.id_token.fetch_id_token",
        lambda request, audience: "mock-id-token",
    )
    client = MCPRouteClient(
        Settings(
            mcp_auth_mode="gcp_id_token",
            mcp_base_url="https://agent-3-mcp.example.com",
        )
    )

    client.estimate_route(
        origin=Coordinates(lat=41.9, lng=12.4),
        destination=Coordinates(lat=41.8, lng=12.5),
        transport_preferences=["walking"],
    )

    assert captured_headers == {"Authorization": "Bearer mock-id-token"}


def test_mcp_route_client_raises_when_id_token_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "Client", lambda timeout: _MockClient())

    def _fail_fetch(request: object, audience: str) -> str:
        raise RuntimeError(f"cannot fetch token for {audience}")

    monkeypatch.setattr("agent3.services.mcp_client.id_token.fetch_id_token", _fail_fetch)
    client = MCPRouteClient(
        Settings(
            mcp_auth_mode="gcp_id_token",
            mcp_base_url="https://agent-3-mcp.example.com",
        )
    )

    with pytest.raises(RouteEstimationError):
        client.estimate_route(
            origin=Coordinates(lat=41.9, lng=12.4),
            destination=Coordinates(lat=41.8, lng=12.5),
            transport_preferences=["walking"],
        )


def test_mcp_route_client_raises_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout: _MockClient(error=httpx.ConnectError("nope")),
    )
    client = MCPRouteClient(Settings())

    with pytest.raises(RouteEstimationError):
        client.estimate_route(
            origin=Coordinates(lat=41.9, lng=12.4),
            destination=Coordinates(lat=41.8, lng=12.5),
            transport_preferences=["walking"],
        )


def test_mcp_route_client_rejects_unsupported_transport_mode() -> None:
    client = MCPRouteClient(Settings())

    with pytest.raises(RouteEstimationError):
        client.estimate_route(
            origin=Coordinates(lat=41.9, lng=12.4),
            destination=Coordinates(lat=41.8, lng=12.5),
            transport_preferences=["scooter"],
        )
