import pytest
from pydantic import ValidationError

from agent3.models.plan import PlanRequest


def test_plan_request_rejects_invalid_day_window() -> None:
    with pytest.raises(ValidationError):
        PlanRequest.model_validate(
            {
                "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
                "day_start": "18:00:00",
                "day_end": "09:00:00",
                "transport_preferences": ["walk"],
                "places": [
                    {
                        "id": "colosseum",
                        "name": "Colosseum",
                        "lat": 41.8902,
                        "lng": 12.4922,
                        "estimated_duration_minutes": 90,
                        "priority": 5,
                    }
                ],
            }
        )


def test_plan_request_normalizes_transport_preferences() -> None:
    request = PlanRequest.model_validate(
        {
            "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
            "day_start": "09:00:00",
            "day_end": "18:00:00",
            "transport_preferences": [" Walk ", "walk", "METRO"],
            "places": [
                {
                    "id": "pantheon",
                    "name": "Pantheon",
                    "lat": 41.8986,
                    "lng": 12.4769,
                    "estimated_duration_minutes": 45,
                    "priority": 4,
                }
            ],
        }
    )

    assert request.transport_preferences == ["walk", "metro"]
