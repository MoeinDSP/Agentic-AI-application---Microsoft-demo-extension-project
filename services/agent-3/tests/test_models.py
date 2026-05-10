import pytest
from pydantic import ValidationError

from agent3.models.plan import DaySchedulingRequest


def _payload() -> dict[str, object]:
    return {
        "day_start": "2026-04-20T09:00:00",
        "day_end": "2026-04-20T18:00:00",
        "food_budget_per_day": 40,
        "preferences": [" Italian ", "italian"],
        "acceptable_transport_modes": [" Walking ", "bicycling"],
        "places": [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "location": {"latitude": 41.8986, "longitude": 12.4769},
                "estimated_visit_duration_minutes": 45,
                "estimated_cost": 5,
                "category": "historical",
                "rating": 4.8,
                "summary": "Ancient Roman temple.",
                "priority_score": 0.9,
                "opening_hours": [
                    {
                        "day_of_week": "Monday",
                        "open_time": "09:00:00",
                        "close_time": "18:00:00",
                    }
                ],
            }
        ],
    }


def test_day_scheduling_request_accepts_pdf_shaped_payload() -> None:
    request = DaySchedulingRequest.model_validate(_payload())

    assert request.day_start.isoformat() == "2026-04-20T09:00:00"
    assert request.preferences == ["italian"]
    assert request.acceptable_transport_modes == ["walking", "bicycling"]
    assert request.places[0].location.latitude == 41.8986
    assert request.places[0].opening_hours[0].day_of_week == "monday"


def test_day_scheduling_request_rejects_invalid_day_window() -> None:
    payload = _payload()
    payload["day_end"] = "2026-04-20T08:00:00"

    with pytest.raises(ValidationError):
        DaySchedulingRequest.model_validate(payload)


def test_day_scheduling_request_rejects_empty_places() -> None:
    payload = _payload()
    payload["places"] = []

    with pytest.raises(ValidationError):
        DaySchedulingRequest.model_validate(payload)


def test_day_scheduling_request_rejects_unsupported_transport_mode() -> None:
    payload = _payload()
    payload["acceptable_transport_modes"] = ["scooter"]

    with pytest.raises(ValidationError):
        DaySchedulingRequest.model_validate(payload)


def test_day_scheduling_request_accepts_all_pdf_transport_modes() -> None:
    payload = _payload()
    payload["acceptable_transport_modes"] = [
        "walking",
        "driving",
        "transit",
        "bicycling",
    ]

    request = DaySchedulingRequest.model_validate(payload)

    assert request.acceptable_transport_modes == [
        "walking",
        "driving",
        "transit",
        "bicycling",
    ]


def test_day_scheduling_request_rejects_invalid_opening_hours() -> None:
    payload = _payload()
    places = payload["places"]
    assert isinstance(places, list)
    place = places[0]
    assert isinstance(place, dict)
    place["opening_hours"] = [
        {
            "day_of_week": "monday",
            "open_time": "18:00:00",
            "close_time": "09:00:00",
        }
    ]

    with pytest.raises(ValidationError):
        DaySchedulingRequest.model_validate(payload)
