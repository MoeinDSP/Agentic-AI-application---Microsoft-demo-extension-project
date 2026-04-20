from fastapi.testclient import TestClient

from agent4.main import app
from agent4.models.meal import Coordinates, MealRecommendationRequest
from agent4.services.recommender import RecommenderService


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
    assert len(payload["candidates"]) == 5
    assert payload["candidates"][0]["id"] == "mercato-panini"


def test_recommender_prefers_budget_fit_then_preference_then_rating() -> None:
    service = RecommenderService()

    response = service.recommend(
        MealRecommendationRequest(
            time_of_day="lunch",
            search_center=Coordinates(lat=41.9, lng=12.48),
            search_radius_meters=1000,
            budget_per_meal_per_person=20,
            preferences=["italian"],
        )
    )

    assert [candidate.id for candidate in response.candidates[:3]] == [
        "mercato-panini",
        "osteria-transit-stop",
        "trattoria-della-luce",
    ]


def test_recommender_returns_no_candidates_when_radius_filters_everything() -> None:
    service = RecommenderService()

    response = service.recommend(
        MealRecommendationRequest(
            time_of_day="lunch",
            search_center=Coordinates(lat=41.9, lng=12.48),
            search_radius_meters=10,
            budget_per_meal_per_person=20,
            preferences=["italian"],
        )
    )

    assert response.candidates == []


def test_recommender_accepts_dinner_requests() -> None:
    service = RecommenderService()

    response = service.recommend(
        MealRecommendationRequest(
            time_of_day="dinner",
            search_center=Coordinates(lat=41.9, lng=12.48),
            search_radius_meters=1000,
            budget_per_meal_per_person=24,
            preferences=["italian"],
        )
    )

    assert response.candidates
    assert response.candidates[0].id == "trattoria-della-luce"
