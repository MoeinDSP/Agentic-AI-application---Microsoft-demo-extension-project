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
        response: _MockResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> "_MockClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict[str, object]) -> _MockResponse:
        self.calls.append((url, json))
        if self._error is not None:
            raise self._error
        return self._response or _MockResponse({})


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
        response=_MockResponse(
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
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    client = Agent4MealClient(Settings())

    response = client.recommend_meal(_request())

    assert mock_client.calls[0][0].endswith("/v1/recommend-meal")
    assert response.candidates[0].id == "mercato-panini"


def test_agent4_client_supports_a2a_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = _MockClient(
        response=_MockResponse(
            {
                "request_id": "req-1",
                "status": "completed",
                "result_type": "meal_recommendations",
                "output": {
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
                },
                "notes": ["minimal_a2a_boundary"],
            }
        )
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: mock_client)
    client = Agent4MealClient(
        Settings(agent4_invocation_mode=AGENT4_INVOCATION_MODE_A2A)
    )

    response = client.recommend_meal(_request())

    assert mock_client.calls[0][0].endswith("/a2a")
    assert mock_client.calls[0][1]["action"] == "recommend_meal"
    assert response.candidates[0].id == "mercato-panini"


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


def test_settings_reject_invalid_agent4_invocation_mode() -> None:
    with pytest.raises(ValueError):
        Settings(agent4_invocation_mode="grpc")


def test_settings_accept_http_invocation_mode() -> None:
    settings = Settings(agent4_invocation_mode=AGENT4_INVOCATION_MODE_HTTP)

    assert settings.agent4_invocation_mode == AGENT4_INVOCATION_MODE_HTTP
