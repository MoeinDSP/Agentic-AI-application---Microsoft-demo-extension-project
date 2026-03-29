from agent3.models.mcp import TravelEstimate
from agent3.models.plan import PlanRequest
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
