from fastapi.testclient import TestClient

from agent3.main import app
from agent3.models.agent4 import MealRecommendationResponse, RestaurantCandidate
from agent3.models.mcp import TravelEstimate
from agent3.services.a2a import A2AService, get_a2a_service
from agent3.services.planner import PlannerService, get_planner_service


class FixedRouteClient:
    def __init__(self, travel_minutes: int) -> None:
        self._travel_minutes = travel_minutes

    def estimate_route(self, **kwargs: object) -> TravelEstimate:
        transport_preferences = kwargs.get("transport_preferences", [])
        mode = transport_preferences[0] if transport_preferences else "walking"
        return TravelEstimate(
            source="mcp",
            mode=mode,
            estimated_duration_minutes=self._travel_minutes,
            notes=["mock_mcp"],
        )


class FixedMealClient:
    def recommend_meal(self, request: object) -> MealRecommendationResponse:
        _ = request
        return MealRecommendationResponse(
            candidates=[
                RestaurantCandidate(
                    id="trattoria-della-luce",
                    name="Trattoria della Luce",
                    location={"lat": 41.8991, "lng": 12.4828},
                    price_level=2,
                    cuisines=["italian", "roman"],
                    rating=4.7,
                    summary="Classic Roman lunch and dinner menu.",
                )
            ]
        )


def _plan_payload() -> dict[str, object]:
    return {
        "day_start": "2026-04-20T09:00:00",
        "day_end": "2026-04-20T15:00:00",
        "food_budget_per_day": 20,
        "preferences": ["italian"],
        "acceptable_transport_modes": ["walking"],
        "places": [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "location": {"latitude": 41.8986, "longitude": 12.4769},
                "estimated_visit_duration_minutes": 45,
                "priority_score": 4,
            },
            {
                "id": "colosseum",
                "name": "Colosseum",
                "location": {"latitude": 41.8902, "longitude": 12.4922},
                "estimated_visit_duration_minutes": 180,
                "priority_score": 5,
            },
        ],
    }


def test_agent_card_endpoint_returns_discovery_metadata() -> None:
    client = TestClient(app)

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "agent-3"
    assert payload["interaction_mode"] == "a2a-ready-http"
    assert payload["auth"]["mode"] == "none"
    assert any(endpoint["path"] == "/a2a" for endpoint in payload["endpoints"])


def test_v1_plan_returns_day_scheduling_result() -> None:
    client = TestClient(app)

    response = client.post("/v1/plan", json=_plan_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["day_schedule"]["date"] == "2026-04-20"
    assert payload["day_schedule"]["events"][0]["event_type"] == "visit"
    assert payload["day_schedule"]["events"][0]["place"]["id"] == "colosseum"
    assert "unscheduled_places" in payload
    assert "warnings" in payload


def test_v1_plan_and_a2a_return_consistent_day_schedules() -> None:
    planner = PlannerService(
        route_client=FixedRouteClient(0),
        meal_client=FixedMealClient(),
    )
    app.dependency_overrides[get_planner_service] = lambda: planner
    app.dependency_overrides[get_a2a_service] = lambda: A2AService(planner)
    client = TestClient(app)

    try:
        plan_response = client.post("/v1/plan", json=_plan_payload())
        a2a_response = client.post(
            "/a2a",
            json={
                "request_id": "req-123",
                "action": "plan_day",
                "input": _plan_payload(),
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert plan_response.status_code == 200
    assert a2a_response.status_code == 200
    assert a2a_response.json()["result_type"] == "day_schedule"
    assert plan_response.json() == a2a_response.json()["output"]


def test_v1_plan_rejects_empty_places_list() -> None:
    client = TestClient(app)
    payload = _plan_payload()
    payload["places"] = []

    response = client.post("/v1/plan", json=payload)

    assert response.status_code == 422


def test_a2a_rejects_unsupported_transport_mode() -> None:
    client = TestClient(app)
    payload = _plan_payload()
    payload["acceptable_transport_modes"] = ["scooter"]

    response = client.post(
        "/a2a",
        json={
            "request_id": "req-unsupported",
            "action": "plan_day",
            "input": payload,
        },
    )

    assert response.status_code == 422


def test_v1_plan_supports_bicycling_transport_mode() -> None:
    client = TestClient(app)
    payload = _plan_payload()
    payload["acceptable_transport_modes"] = ["bicycling"]

    response = client.post("/v1/plan", json=payload)

    assert response.status_code == 200
