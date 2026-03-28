from agent3.models.plan import PlanRequest
from agent3.services.planner import PlannerService


def _build_request(places: list[dict[str, object]]) -> PlanRequest:
    return PlanRequest.model_validate(
        {
            "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
            "day_start": "09:00:00",
            "day_end": "12:00:00",
            "transport_preferences": ["walk"],
            "places": places,
        }
    )


def test_planner_schedules_all_places_when_they_fit() -> None:
    request = _build_request(
        [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            },
            {
                "id": "trevi",
                "name": "Trevi Fountain",
                "lat": 41.9009,
                "lng": 12.4833,
                "estimated_duration_minutes": 30,
                "priority": 3,
            },
        ]
    )

    response = PlannerService().plan_day(request)

    assert [stop.place_id for stop in response.ordered_stops] == ["pantheon", "trevi"]
    assert response.ordered_stops[0].start_time.isoformat() == "09:00:00"
    assert response.ordered_stops[0].end_time.isoformat() == "09:45:00"
    assert response.ordered_stops[1].start_time.isoformat() == "09:45:00"
    assert response.ordered_stops[1].end_time.isoformat() == "10:15:00"
    assert response.dropped_places == []
    assert response.feasibility is True


def test_planner_drops_remaining_places_when_only_some_fit() -> None:
    request = _build_request(
        [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 60,
                "priority": 4,
            },
            {
                "id": "colosseum",
                "name": "Colosseum",
                "lat": 41.8902,
                "lng": 12.4922,
                "estimated_duration_minutes": 120,
                "priority": 5,
            },
            {
                "id": "forum",
                "name": "Roman Forum",
                "lat": 41.8925,
                "lng": 12.4853,
                "estimated_duration_minutes": 90,
                "priority": 3,
            },
        ]
    )

    response = PlannerService().plan_day(request)

    assert [stop.place_id for stop in response.ordered_stops] == ["colosseum", "pantheon"]
    assert [place.place_id for place in response.dropped_places] == ["forum"]
    assert response.dropped_places[0].reason == "insufficient_time"
    assert response.feasibility is True


def test_planner_marks_infeasible_when_no_places_fit() -> None:
    request = _build_request(
        [
            {
                "id": "museum",
                "name": "Vatican Museums",
                "lat": 41.9065,
                "lng": 12.4536,
                "estimated_duration_minutes": 240,
                "priority": 5,
            }
        ]
    )

    response = PlannerService().plan_day(request)

    assert response.ordered_stops == []
    assert response.dropped_places[0].reason == "insufficient_time"
    assert response.feasibility is False
