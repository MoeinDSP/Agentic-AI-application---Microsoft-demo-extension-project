from fastapi.testclient import TestClient

from agent3.main import app


def _plan_payload() -> dict[str, object]:
    return {
        "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
        "day_start": "09:00:00",
        "day_end": "11:00:00",
        "transport_preferences": ["walk"],
        "places": [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            },
            {
                "id": "colosseum",
                "name": "Colosseum",
                "lat": 41.8902,
                "lng": 12.4922,
                "estimated_duration_minutes": 90,
                "priority": 5,
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


def test_a2a_request_returns_deterministic_plan_response() -> None:
    client = TestClient(app)

    response = client.post(
        "/a2a",
        json={
            "request_id": "req-123",
            "action": "plan_day",
            "input": _plan_payload(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req-123"
    assert payload["status"] == "completed"
    assert payload["result_type"] == "plan"
    assert payload["output"]["ordered_stops"][0]["place_id"] == "colosseum"
    assert payload["output"]["ordered_stops"][0]["start_time"] == "09:00:00"
    assert payload["output"]["ordered_stops"][0]["end_time"] == "10:30:00"
    assert payload["output"]["dropped_places"][0]["place_id"] == "pantheon"
    assert payload["output"]["dropped_places"][0]["reason"] == "insufficient_time"
    assert payload["output"]["feasibility"] is True
    assert "minimal_a2a_boundary" in payload["notes"]


def test_a2a_request_rejects_malformed_payload() -> None:
    client = TestClient(app)

    response = client.post(
        "/a2a",
        json={
            "request_id": "req-456",
            "action": "plan_day",
            "input": {
                "start_location": {"lat": 41.9028, "lng": 12.4964},
                "day_start": "11:00:00",
                "day_end": "09:00:00",
                "transport_preferences": ["walk"],
                "places": [],
            },
        },
    )

    assert response.status_code == 422


def test_v1_plan_and_a2a_return_consistent_plans() -> None:
    client = TestClient(app)

    plan_response = client.post("/v1/plan", json=_plan_payload())
    a2a_response = client.post(
        "/a2a",
        json={
            "request_id": "req-789",
            "action": "plan_day",
            "input": _plan_payload(),
        },
    )

    assert plan_response.status_code == 200
    assert a2a_response.status_code == 200
    assert plan_response.json() == a2a_response.json()["output"]


def test_v1_plan_rejects_empty_places_list() -> None:
    client = TestClient(app)
    payload = _plan_payload()
    payload["places"] = []

    response = client.post("/v1/plan", json=payload)

    assert response.status_code == 422


def test_a2a_rejects_empty_places_list() -> None:
    client = TestClient(app)
    payload = _plan_payload()
    payload["places"] = []

    response = client.post(
        "/a2a",
        json={
            "request_id": "req-empty",
            "action": "plan_day",
            "input": payload,
        },
    )

    assert response.status_code == 422
