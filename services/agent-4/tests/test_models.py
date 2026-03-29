import pytest
from pydantic import ValidationError

from agent4.models.meal import MealRecommendationRequest


def test_request_normalizes_preferences() -> None:
    request = MealRecommendationRequest.model_validate(
        {
            "time_of_day": "lunch",
            "search_center": {"lat": 41.9, "lng": 12.48},
            "search_radius_meters": 1000,
            "budget_per_meal_per_person": 20,
            "preferences": [" Italian ", "vegetarian"],
        }
    )

    assert request.preferences == ["italian", "vegetarian"]


def test_request_rejects_non_positive_radius() -> None:
    with pytest.raises(ValidationError):
        MealRecommendationRequest.model_validate(
            {
                "time_of_day": "lunch",
                "search_center": {"lat": 41.9, "lng": 12.48},
                "search_radius_meters": 0,
                "preferences": [],
            }
        )
