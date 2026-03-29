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

    def post(self, url: str, json: dict[str, object]) -> _MockResponse:
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
                    "mode": "walk",
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
        transport_preferences=["walk"],
    )

    assert response.source == "mcp"
    assert response.estimated_duration_minutes == 14
    assert response.notes == ["mock_success"]


def test_mcp_route_client_uses_default_transport_mode_when_preferences_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturingClient(_MockClient):
        def post(self, url: str, json: dict[str, object]) -> _MockResponse:
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

    assert captured["mode"] == "walk"
    assert response.mode == "walk"


def test_mcp_route_client_uses_explicit_transport_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturingClient(_MockClient):
        def post(self, url: str, json: dict[str, object]) -> _MockResponse:
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
        transport_preferences=["drive"],
    )

    assert captured["mode"] == "drive"
    assert response.mode == "drive"


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
            transport_preferences=["walk"],
        )


def test_mcp_route_client_rejects_unsupported_transport_mode() -> None:
    client = MCPRouteClient(Settings())

    with pytest.raises(RouteEstimationError):
        client.estimate_route(
            origin=Coordinates(lat=41.9, lng=12.4),
            destination=Coordinates(lat=41.8, lng=12.5),
            transport_preferences=["scooter"],
        )
