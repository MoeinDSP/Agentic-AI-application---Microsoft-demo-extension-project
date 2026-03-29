from fastapi.testclient import TestClient

from agent4.main import app


def test_recommend_meal_returns_candidates() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/recommend-meal",
        json={
            "time_of_day": "lunch",
            "search_center": {"lat": 41.9, "lng": 12.48},
            "search_radius_meters": 1000,
            "budget_per_meal_per_person": 20,
            "preferences": ["italian"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["candidates"]) == 3
    assert payload["candidates"][0]["id"] == "trattoria-della-luce"
