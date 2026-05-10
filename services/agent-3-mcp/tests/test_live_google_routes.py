import pytest
from fastapi.testclient import TestClient

from agent3_mcp.core.config import Settings
from agent3_mcp.main import create_app


def test_live_route_estimate_calls_google_routes() -> None:
    settings = Settings()
    if not settings.google_maps_api_key:
        pytest.skip("AGENT3_MCP_GOOGLE_MAPS_API_KEY is not configured")

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/tools/route-estimate",
            json={
                "origin": {"lat": 41.9028, "lng": 12.4964},
                "destination": {"lat": 41.8902, "lng": 12.4922},
                "mode": "transit",
                "departure_time": "2026-05-08T09:00:00Z",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "transit"
    assert payload["estimated_distance_km"] > 0
    assert payload["estimated_duration_minutes"] > 0
    assert "provider=google_routes" in payload["notes"]
