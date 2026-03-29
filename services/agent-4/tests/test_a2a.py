from fastapi.testclient import TestClient

from agent4.main import app


def _meal_payload() -> dict[str, object]:
    return {
        "time_of_day": "lunch",
        "search_center": {"lat": 41.9, "lng": 12.48},
        "search_radius_meters": 1000,
        "budget_per_meal_per_person": 20,
        "preferences": ["italian"],
    }


def test_agent_card_endpoint_returns_discovery_metadata() -> None:
    client = TestClient(app)

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "agent-4"
    assert payload["interaction_mode"] == "a2a-ready-http"
    assert payload["auth"]["mode"] == "none"
    assert any(endpoint["path"] == "/a2a" for endpoint in payload["endpoints"])


def test_a2a_request_returns_recommendation_output() -> None:
    client = TestClient(app)

    response = client.post(
        "/a2a",
        json={
            "request_id": "req-123",
            "action": "recommend_meal",
            "input": _meal_payload(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req-123"
    assert payload["status"] == "completed"
    assert payload["result_type"] == "meal_recommendations"
    assert payload["output"]["candidates"][0]["id"] == "mercato-panini"
    assert "minimal_a2a_boundary" in payload["notes"]


def test_a2a_request_rejects_invalid_action() -> None:
    client = TestClient(app)

    response = client.post(
        "/a2a",
        json={
            "request_id": "req-456",
            "action": "plan_day",
            "input": _meal_payload(),
        },
    )

    assert response.status_code == 422
