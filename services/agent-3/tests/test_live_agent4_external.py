import httpx
import pytest

from agent3.core.config import Settings
from agent3.models.agent4 import MealRecommendationRequest
from agent3.models.plan import Coordinates
from agent3.services.agent4_client import Agent4MealClient


def test_live_agent4_recommendation_returns_candidates() -> None:
    base_settings = Settings()
    if not base_settings.agent4_base_url:
        pytest.skip("AGENT3_AGENT4_BASE_URL is not configured")
    try:
        with httpx.Client(timeout=5) as probe_client:
            probe_response = probe_client.get(
                f"{base_settings.agent4_base_url.rstrip('/')}/.well-known/agent-card.json"
            )
            probe_response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"external Agent 4 endpoint is unreachable: {exc}")

    settings = Settings(
        agent4_base_url=base_settings.agent4_base_url,
        agent4_invocation_mode=base_settings.agent4_invocation_mode,
        agent4_timeout_seconds=10,
        agent4_poll_interval_seconds=1.0,
        agent4_max_wait_seconds=30,
    )
    client = Agent4MealClient(settings)
    request = MealRecommendationRequest(
        time_of_day="lunch",
        search_center=Coordinates(lat=41.9028, lng=12.4964),
        search_radius_meters=1200,
        budget_per_meal_per_person=25,
        preferences=["italian"],
    )

    response = client.recommend_meal(request)

    assert response.candidates
    candidate = response.candidates[0]
    assert candidate.id
    assert candidate.name
    assert candidate.rating >= 0
    assert candidate.price_level >= 1
