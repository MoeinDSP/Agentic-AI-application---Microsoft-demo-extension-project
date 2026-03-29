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
            "transport_preferences": [" Walk ", "walk", "DRIVE"],
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

    assert request.transport_preferences == ["walk", "drive"]


def test_plan_request_accepts_optional_opening_hours() -> None:
    request = PlanRequest.model_validate(
        {
            "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
            "day_start": "09:00:00",
            "day_end": "18:00:00",
            "transport_preferences": ["walk"],
            "places": [
                {
                    "id": "pantheon",
                    "name": "Pantheon",
                    "lat": 41.8986,
                    "lng": 12.4769,
                    "estimated_duration_minutes": 45,
                    "priority": 4,
                    "opens_at": "10:00:00",
                    "closes_at": "17:00:00",
                }
            ],
        }
    )

    assert request.places[0].opens_at is not None
    assert request.places[0].opens_at.isoformat() == "10:00:00"
    assert request.places[0].closes_at is not None
    assert request.places[0].closes_at.isoformat() == "17:00:00"


def test_plan_request_rejects_invalid_opening_window() -> None:
    with pytest.raises(ValidationError):
        PlanRequest.model_validate(
            {
                "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
                "day_start": "09:00:00",
                "day_end": "18:00:00",
                "transport_preferences": ["walk"],
                "places": [
                    {
                        "id": "pantheon",
                        "name": "Pantheon",
                        "lat": 41.8986,
                        "lng": 12.4769,
                        "estimated_duration_minutes": 45,
                        "priority": 4,
                        "opens_at": "17:00:00",
                        "closes_at": "10:00:00",
                    }
                ],
            }
        )


def test_plan_request_supports_lunch_configuration() -> None:
    request = PlanRequest.model_validate(
        {
            "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
            "day_start": "09:00:00",
            "day_end": "18:00:00",
            "transport_preferences": ["walk"],
            "lunch_required": True,
            "lunch_time_window_start": "12:30:00",
            "lunch_time_window_end": "14:30:00",
            "lunch_duration_minutes": 45,
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

    assert request.lunch_required is True
    assert request.lunch_time_window_start.isoformat() == "12:30:00"
    assert request.lunch_time_window_end.isoformat() == "14:30:00"
    assert request.lunch_duration_minutes == 45


def test_plan_request_rejects_invalid_lunch_window() -> None:
    with pytest.raises(ValidationError):
        PlanRequest.model_validate(
            {
                "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
                "day_start": "09:00:00",
                "day_end": "18:00:00",
                "transport_preferences": ["walk"],
                "lunch_required": True,
                "lunch_time_window_start": "14:00:00",
                "lunch_time_window_end": "12:00:00",
                "lunch_duration_minutes": 45,
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
