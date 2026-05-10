from datetime import datetime

import httpx
import pytest

from agent3_mcp.core.config import Settings
from agent3_mcp.models.tools import Coordinates, RouteEstimateRequest
from agent3_mcp.services.tools import FIELD_MASK, GOOGLE_ROUTES_URL, ToolService


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
        self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def __enter__(self) -> "_MockClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(
        self,
        url: str,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> _MockResponse:
        self.calls.append((url, json, headers))
        if self._error is not None:
            raise self._error
        return self._response or _MockResponse({})


def test_route_estimate_uses_google_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = _MockClient(
        response=_MockResponse(
            {
                "routes": [
                    {
                        "distanceMeters": 1250,
                        "duration": "840s",
                        "warnings": ["walk and bicycle routes are in beta"],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    service = ToolService(
        Settings(
            google_maps_api_key="test-key",
            google_routes_timeout_seconds=5,
        )
    )

    response = service.estimate_route(
        RouteEstimateRequest(
            origin=Coordinates(lat=41.9028, lng=12.4964),
            destination=Coordinates(lat=41.8902, lng=12.4922),
            mode="walking",
        )
    )

    assert mock_client.calls[0][0] == GOOGLE_ROUTES_URL
    assert mock_client.calls[0][2]["X-Goog-Api-Key"] == "test-key"
    assert mock_client.calls[0][2]["X-Goog-FieldMask"] == FIELD_MASK
    assert mock_client.calls[0][1]["travelMode"] == "WALK"
    assert response.mode == "walking"
    assert response.estimated_distance_km == 1.25
    assert response.estimated_duration_minutes == 14
    assert "provider=google_routes" in response.notes


def test_route_estimate_strips_whitespace_from_google_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = _MockClient(
        response=_MockResponse(
            {
                "routes": [
                    {
                        "distanceMeters": 1250,
                        "duration": "840s",
                        "warnings": [],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    service = ToolService(
        Settings(
            google_maps_api_key="test-key\r\n",
            google_routes_timeout_seconds=5,
        )
    )

    service.estimate_route(
        RouteEstimateRequest(
            origin=Coordinates(lat=41.9028, lng=12.4964),
            destination=Coordinates(lat=41.8902, lng=12.4922),
            mode="walking",
        )
    )

    assert mock_client.calls[0][2]["X-Goog-Api-Key"] == "test-key"


def test_route_estimate_uses_transit_departure_time(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = _MockClient(
        response=_MockResponse(
            {
                "routes": [
                    {
                        "distanceMeters": 2200,
                        "duration": "600s",
                        "warnings": [],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    service = ToolService(Settings(google_maps_api_key="test-key"))
    departure_time = datetime.fromisoformat("2026-05-02T08:15:00")

    response = service.estimate_route(
        RouteEstimateRequest(
            origin=Coordinates(lat=41.9028, lng=12.4964),
            destination=Coordinates(lat=41.8902, lng=12.4922),
            mode="transit",
            departure_time=departure_time,
        )
    )

    assert mock_client.calls[0][1]["departureTime"] == "2026-05-02T08:15:00"
    assert response.estimated_duration_minutes == 10


def test_route_estimate_forwards_departure_time_for_non_transit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = _MockClient(
        response=_MockResponse(
            {
                "routes": [
                    {
                        "distanceMeters": 2200,
                        "duration": "600s",
                        "warnings": [],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    service = ToolService(Settings(google_maps_api_key="test-key"))

    service.estimate_route(
        RouteEstimateRequest(
            origin=Coordinates(lat=41.9028, lng=12.4964),
            destination=Coordinates(lat=41.8902, lng=12.4922),
            mode="driving",
            departure_time=datetime.fromisoformat("2026-05-02T08:15:00"),
        )
    )

    assert mock_client.calls[0][1]["departureTime"] == "2026-05-02T08:15:00"


def test_route_estimate_requires_google_api_key() -> None:
    service = ToolService(Settings(google_maps_api_key=None))

    with pytest.raises(ValueError):
        service.estimate_route(
            RouteEstimateRequest(
                origin=Coordinates(lat=41.9028, lng=12.4964),
                destination=Coordinates(lat=41.8902, lng=12.4922),
                mode="driving",
            )
        )


def test_route_estimate_raises_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout: _MockClient(error=httpx.ConnectError("nope")),
    )
    service = ToolService(Settings(google_maps_api_key="test-key"))

    with pytest.raises(httpx.ConnectError):
        service.estimate_route(
            RouteEstimateRequest(
                origin=Coordinates(lat=41.9028, lng=12.4964),
                destination=Coordinates(lat=41.8902, lng=12.4922),
                mode="driving",
            )
        )


def test_route_estimate_raises_on_invalid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = _MockClient(response=_MockResponse({"routes": []}))
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    service = ToolService(Settings(google_maps_api_key="test-key"))

    with pytest.raises(ValueError):
        service.estimate_route(
            RouteEstimateRequest(
                origin=Coordinates(lat=41.9028, lng=12.4964),
                destination=Coordinates(lat=41.8902, lng=12.4922),
                mode="bicycling",
            )
        )
