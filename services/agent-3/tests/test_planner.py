from datetime import time

from agent3.models.mcp import TravelEstimate
from agent3.models.plan import (
    LUNCH_INSERTED_NOTE,
    LUNCH_NOT_INSERTED_NOTE,
    STOP_TYPE_LUNCH,
    STOP_TYPE_PLACE,
    PlanRequest,
)
from agent3.services.planner import PlannerService


class FixedRouteClient:
    def __init__(self, travel_minutes: int) -> None:
        self._travel_minutes = travel_minutes

    def estimate_route(self, **kwargs: object) -> TravelEstimate:
        transport_preferences = kwargs.get("transport_preferences", [])
        mode = transport_preferences[0] if transport_preferences else "walk"
        return TravelEstimate(
            source="mcp",
            mode=mode,
            estimated_duration_minutes=self._travel_minutes,
            notes=["mock_mcp"],
        )


class FailingRouteClient:
    def estimate_route(self, **_: object) -> TravelEstimate:
        from agent3.services.mcp_client import RouteEstimationError

        raise RouteEstimationError("boom")


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


def test_planner_accounts_for_travel_time_when_all_places_fit() -> None:
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

    response = PlannerService(FixedRouteClient(10)).plan_day(request)

    assert [stop.place_id for stop in response.ordered_stops] == ["pantheon", "trevi"]
    assert response.ordered_stops[0].arrival_time.isoformat() == "09:10:00"
    assert response.ordered_stops[0].start_time.isoformat() == "09:10:00"
    assert response.ordered_stops[0].end_time.isoformat() == "09:55:00"
    assert response.ordered_stops[0].travel_minutes_from_previous == 10
    assert response.ordered_stops[1].arrival_time.isoformat() == "10:05:00"
    assert response.ordered_stops[1].end_time.isoformat() == "10:35:00"
    assert response.total_travel_minutes == 20
    assert response.total_visit_minutes == 75
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


def test_planner_drops_place_when_travel_time_pushes_it_past_day_end() -> None:
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
                "id": "colosseum",
                "name": "Colosseum",
                "lat": 41.8902,
                "lng": 12.4922,
                "estimated_duration_minutes": 90,
                "priority": 5,
            },
        ]
    )

    response = PlannerService(FixedRouteClient(30)).plan_day(request)

    assert [stop.place_id for stop in response.ordered_stops] == ["colosseum"]
    assert [place.place_id for place in response.dropped_places] == ["pantheon"]
    assert response.total_travel_minutes == 30
    assert response.total_visit_minutes == 90


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


def test_planner_uses_fallback_minutes_when_route_estimation_fails() -> None:
    request = _build_request(
        [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            }
        ]
    )

    response = PlannerService(
        route_client=FailingRouteClient(),
        fallback_travel_minutes=12,
    ).plan_day(request)

    assert response.ordered_stops[0].arrival_time.isoformat() == "09:12:00"
    assert response.ordered_stops[0].travel_minutes_from_previous == 12
    assert response.total_travel_minutes == 12
    assert "travel_time_source=fallback" in response.notes
    assert "fallback_travel_minutes=12" in response.notes


def test_planner_defaults_to_walk_when_transport_preferences_missing() -> None:
    request = _build_request(
        [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            }
        ]
    )
    request.transport_preferences = []

    response = PlannerService(FixedRouteClient(10)).plan_day(request)

    assert response.selected_transport_mode == "walk"
    assert "selected_transport_mode=walk" in response.notes


def test_planner_uses_explicit_drive_mode() -> None:
    request = _build_request(
        [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            }
        ]
    )
    request.transport_preferences = ["drive"]

    response = PlannerService(FixedRouteClient(5)).plan_day(request)

    assert response.selected_transport_mode == "drive"
    assert "selected_transport_mode=drive" in response.notes


def test_planner_treats_missing_opening_hours_as_unrestricted() -> None:
    request = _build_request(
        [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            }
        ]
    )

    response = PlannerService(FixedRouteClient(10)).plan_day(request)

    assert [stop.place_id for stop in response.ordered_stops] == ["pantheon"]
    assert response.dropped_places == []


def test_planner_drops_place_when_arrival_is_before_opening_time() -> None:
    request = _build_request(
        [
            {
                "id": "colosseum",
                "name": "Colosseum",
                "lat": 41.8902,
                "lng": 12.4922,
                "estimated_duration_minutes": 90,
                "priority": 5,
                "opens_at": "10:00:00",
                "closes_at": "18:00:00",
            }
        ]
    )

    response = PlannerService(FixedRouteClient(15)).plan_day(request)

    assert response.ordered_stops == []
    assert response.dropped_places[0].place_id == "colosseum"
    assert response.dropped_places[0].reason == "closed_at_arrival"
    assert response.feasibility is False


def test_planner_drops_place_when_visit_would_end_after_closing_time() -> None:
    request = _build_request(
        [
            {
                "id": "colosseum",
                "name": "Colosseum",
                "lat": 41.8902,
                "lng": 12.4922,
                "estimated_duration_minutes": 90,
                "priority": 5,
                "opens_at": "09:00:00",
                "closes_at": "10:00:00",
            }
        ]
    )

    response = PlannerService(FixedRouteClient(15)).plan_day(request)

    assert response.ordered_stops == []
    assert response.dropped_places[0].place_id == "colosseum"
    assert response.dropped_places[0].reason == "closes_before_visit_ends"
    assert response.feasibility is False


def test_planner_leaves_itinerary_unchanged_when_lunch_is_not_requested() -> None:
    request = _build_request(
        [
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            }
        ]
    )
    request.day_end = time(hour=15, minute=0)

    response = PlannerService(FixedRouteClient(10)).build_plan(request)

    assert [stop.stop_type for stop in response.ordered_stops] == [STOP_TYPE_PLACE]
    assert LUNCH_INSERTED_NOTE not in response.notes
    assert LUNCH_NOT_INSERTED_NOTE not in response.notes


def test_planner_inserts_lunch_stop_at_earliest_feasible_window_point() -> None:
    request = _build_request(
        [
            {
                "id": "colosseum",
                "name": "Colosseum",
                "lat": 41.8902,
                "lng": 12.4922,
                "estimated_duration_minutes": 180,
                "priority": 5,
            },
            {
                "id": "pantheon",
                "name": "Pantheon",
                "lat": 41.8986,
                "lng": 12.4769,
                "estimated_duration_minutes": 45,
                "priority": 4,
            },
        ]
    )
    request.day_end = time(hour=15, minute=0)
    request.lunch_required = True
    request.lunch_time_window_start = time(hour=12, minute=0)
    request.lunch_time_window_end = time(hour=14, minute=0)
    request.lunch_duration_minutes = 30

    response = PlannerService().build_plan(request)

    assert [stop.place_id for stop in response.ordered_stops] == [
        "colosseum",
        "lunch",
        "pantheon",
    ]
    assert response.ordered_stops[1].stop_type == STOP_TYPE_LUNCH
    assert response.ordered_stops[1].start_time.isoformat() == "12:00:00"
    assert response.ordered_stops[1].end_time.isoformat() == "12:30:00"
    assert LUNCH_INSERTED_NOTE in response.notes


def test_planner_adds_note_when_required_lunch_cannot_fit() -> None:
    request = _build_request(
        [
            {
                "id": "colosseum",
                "name": "Colosseum",
                "lat": 41.8902,
                "lng": 12.4922,
                "estimated_duration_minutes": 180,
                "priority": 5,
            }
        ]
    )
    request.day_end = time(hour=12, minute=30)
    request.lunch_required = True
    request.lunch_time_window_start = time(hour=12, minute=0)
    request.lunch_time_window_end = time(hour=13, minute=0)
    request.lunch_duration_minutes = 45

    response = PlannerService().build_plan(request)

    assert [stop.place_id for stop in response.ordered_stops] == ["colosseum"]
    assert all(stop.stop_type != STOP_TYPE_LUNCH for stop in response.ordered_stops)
    assert LUNCH_NOT_INSERTED_NOTE in response.notes
