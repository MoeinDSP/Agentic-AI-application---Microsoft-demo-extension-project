import time

from fastapi.testclient import TestClient

from agent3.main import create_app
from agent3.models.agent4 import MealRecommendationResponse, RestaurantCandidate
from agent3.models.mcp import TravelEstimate
from agent3.services.planner import PlannerService


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


def _build_client() -> TestClient:
    planner = PlannerService(
        route_client=FixedRouteClient(0),
        meal_client=FixedMealClient(),
    )
    return TestClient(create_app(planner=planner))


def _send_message(client: TestClient, payload: dict[str, object]) -> dict[str, object]:
    response = client.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "req-send",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-123",
                    "role": "user",
                    "kind": "message",
                    "parts": [
                        {
                            "kind": "data",
                            "data": payload,
                        }
                    ],
                },
                "configuration": {
                    "acceptedOutputModes": ["application/json"],
                },
            },
        },
    )
    assert response.status_code == 200
    return response.json()


def _poll_task(
    client: TestClient,
    task_id: str,
    *,
    max_attempts: int = 50,
) -> dict[str, object]:
    for attempt in range(max_attempts):
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": f"req-get-{attempt}",
                "method": "tasks/get",
                "params": {"id": task_id},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        task = payload["result"]
        if task["status"]["state"] in {"completed", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("task did not reach a terminal state")


def test_health_endpoint_returns_ok() -> None:
    with _build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_agent_card_endpoint_returns_fasta2a_metadata() -> None:
    with _build_client() as client:
        response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Daily Scheduler Agent"
    assert payload["url"] == "http://127.0.0.1:8080"
    assert payload["protocolVersion"] == "0.3.0"
    assert payload["defaultInputModes"] == ["application/json"]
    assert payload["defaultOutputModes"] == ["application/json"]
    assert payload["skills"][0]["id"] == "day-scheduling"
    assert payload["capabilities"]["streaming"] is False


def test_message_send_and_tasks_get_return_completed_day_schedule() -> None:
    with _build_client() as client:
        send_payload = _send_message(client, _plan_payload())
        task_id = send_payload["result"]["id"]
        get_payload = _poll_task(client, task_id)

    assert send_payload["result"]["status"]["state"] == "submitted"
    result_task = get_payload["result"]
    assert result_task["status"]["state"] == "completed"
    artifact = result_task["artifacts"][0]["parts"][0]["data"]
    assert artifact["day_schedule"]["date"] == "2026-04-20"
    assert artifact["day_schedule"]["events"][0]["event_type"] == "visit"
    assert artifact["day_schedule"]["events"][0]["place"]["id"] == "colosseum"
    assert "warnings" in artifact


def test_message_send_marks_task_failed_for_invalid_payload() -> None:
    payload = _plan_payload()
    payload["acceptable_transport_modes"] = ["scooter"]

    with _build_client() as client:
        send_payload = _send_message(client, payload)
        task_id = send_payload["result"]["id"]
        get_payload = _poll_task(client, task_id)

    assert get_payload["result"]["status"]["state"] == "failed"


def test_message_send_supports_bicycling_transport_mode() -> None:
    payload = _plan_payload()
    payload["acceptable_transport_modes"] = ["bicycling"]

    with _build_client() as client:
        send_payload = _send_message(client, payload)
        task_id = send_payload["result"]["id"]
        get_payload = _poll_task(client, task_id)

    assert send_payload["result"]["status"]["state"] == "submitted"
    assert get_payload["result"]["status"]["state"] == "completed"
