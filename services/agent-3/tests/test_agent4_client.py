import httpx
import pytest

from agent3.core.config import Settings
from agent3.models.agent4 import (
    AGENT4_INVOCATION_MODE_A2A,
    AGENT4_INVOCATION_MODE_HTTP,
    MealRecommendationRequest,
)
from agent3.services.agent4_client import Agent4MealClient, MealRecommendationError


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
        responses: list[_MockResponse] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = responses or []
        self._error = error
        self.calls: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

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
        self.calls.append((url, json, headers))
        if self._error is not None:
            raise self._error
        if not self._responses:
            return _MockResponse({})
        return self._responses.pop(0)


def _request() -> MealRecommendationRequest:
    return MealRecommendationRequest.model_validate(
        {
            "time_of_day": "lunch",
            "search_center": {"lat": 41.9, "lng": 12.48},
            "search_radius_meters": 1000,
            "budget_per_meal_per_person": 20,
            "preferences": ["italian"],
        }
    )


def test_agent4_client_defaults_to_http_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = _MockClient(
        responses=[
            _MockResponse(
                {
                    "candidates": [
                        {
                            "id": "mercato-panini",
                            "name": "Mercato Panini",
                            "location": {"lat": 41.8978, "lng": 12.4851},
                            "price_level": 1,
                            "cuisines": ["sandwiches", "italian", "street-food"],
                            "rating": 4.3,
                            "summary": "Budget-friendly panini and takeaway lunch options.",
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    client = Agent4MealClient(
        Settings(
            agent4_invocation_mode=AGENT4_INVOCATION_MODE_HTTP,
            agent4_base_url="http://test-agent4",
        )
    )

    response = client.recommend_meal(_request())

    assert mock_client.calls[0][0].endswith("/v1/recommend-meal")
    assert response.candidates[0].id == "mercato-panini"


def test_agent4_client_supports_real_a2a_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = _MockClient(
        responses=[
            _MockResponse(
                {
                    "jsonrpc": "2.0",
                    "id": "send-1",
                    "result": {"id": "task-123"},
                }
            ),
            _MockResponse(
                {
                    "jsonrpc": "2.0",
                    "id": "poll-1",
                    "result": {
                        "status": {"state": "completed"},
                        "artifacts": [
                            {
                                "artifactId": "result-1",
                                "parts": [
                                    {
                                        "kind": "data",
                                        "data": {
                                            "restaurants": [
                                                {
                                                    "id": "place-1",
                                                    "name": "La Nuova Piazzetta Navona",
                                                    "location": {
                                                        "latitude": 41.8977319,
                                                        "longitude": 12.4735221,
                                                        "address": (
                                                            "Via della Posta Vecchia, 4, Roma"
                                                        ),
                                                    },
                                                    "price_level": 2.0,
                                                    "cuisines": ["Bar"],
                                                    "rating": 4.9,
                                                    "summary": None,
                                                }
                                            ]
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    client = Agent4MealClient(
        Settings(
            agent4_invocation_mode=AGENT4_INVOCATION_MODE_A2A,
            agent4_base_url="http://test-agent4",
            agent4_poll_interval_seconds=0.01,
            agent4_max_wait_seconds=1,
        )
    )

    response = client.recommend_meal(_request())

    assert mock_client.calls[0][0].endswith("/")
    assert mock_client.calls[0][1]["method"] == "message/send"
    assert mock_client.calls[1][1]["method"] == "tasks/get"
    assert response.candidates[0].id == "place-1"
    assert response.candidates[0].summary == "Via della Posta Vecchia, 4, Roma"


def test_agent4_client_raises_on_a2a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout: _MockClient(error=httpx.ConnectError("nope")),
    )
    client = Agent4MealClient(
        Settings(agent4_invocation_mode=AGENT4_INVOCATION_MODE_A2A)
    )

    with pytest.raises(MealRecommendationError):
        client.recommend_meal(_request())


def test_agent4_client_raises_on_a2a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = _MockClient(
        responses=[
            _MockResponse(
                {
                    "jsonrpc": "2.0",
                    "id": "send-1",
                    "result": {"id": "task-123"},
                }
            ),
            _MockResponse(
                {
                    "jsonrpc": "2.0",
                    "id": "poll-1",
                    "result": {
                        "status": {"state": "working"},
                        "artifacts": [],
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    client = Agent4MealClient(
        Settings(
            agent4_invocation_mode=AGENT4_INVOCATION_MODE_A2A,
            agent4_base_url="http://test-agent4",
            agent4_poll_interval_seconds=0.01,
            agent4_max_wait_seconds=0.001,
        )
    )

    with pytest.raises(MealRecommendationError):
        client.recommend_meal(_request())


def test_settings_reject_invalid_agent4_invocation_mode() -> None:
    with pytest.raises(ValueError):
        Settings(agent4_invocation_mode="grpc")


def test_settings_accept_http_invocation_mode() -> None:
    settings = Settings(agent4_invocation_mode=AGENT4_INVOCATION_MODE_HTTP)

    assert settings.agent4_invocation_mode == AGENT4_INVOCATION_MODE_HTTP
