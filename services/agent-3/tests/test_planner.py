from agent3.models.agent4 import MealRecommendationResponse, RestaurantCandidate
from agent3.models.mcp import TravelEstimate
from agent3.models.plan import (
    EVENT_TYPE_MEAL,
    EVENT_TYPE_TRAVEL,
    EVENT_TYPE_VISIT,
    WARNING_AGENT4_UNAVAILABLE,
    WARNING_FALLBACK_TRAVEL,
    WARNING_MEAL_NOT_INSERTED,
    WARNING_NO_RESTAURANT_CANDIDATE,
    WARNING_PLACE_UNSCHEDULED,
    DaySchedulingRequest,
)
from agent3.services.agent4_client import MealRecommendationError
from agent3.services.planner import PlannerService


class FixedRouteClient:
    def __init__(self, travel_minutes: int) -> None:
        self._travel_minutes = travel_minutes

    def estimate_route(self, **kwargs: object) -> TravelEstimate:
        transport_preferences = kwargs.get("transport_preferences", [])
        mode = transport_preferences[0] if transport_preferences else "walking"
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


class CapturingMealClient:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def recommend_meal(self, request: object) -> MealRecommendationResponse:
        self.requests.append(request)
        return MealRecommendationResponse(
            candidates=[
                RestaurantCandidate(
                    id="trattoria-della-luce",
                    name="Trattoria della Luce",
                    location={"lat": 41.8991, "lng": 12.4828},
                    price_level=2,
                    cuisines=["italian", "roman"],
                    rating=4.7,
                    summary="Classic Roman lunch and dinner menu.",
                )
            ]
        )


class EmptyMealClient:
    def recommend_meal(self, request: object) -> MealRecommendationResponse:
        _ = request
        return MealRecommendationResponse(candidates=[])


class FailingMealClient:
    def recommend_meal(self, request: object) -> MealRecommendationResponse:
        _ = request
        raise MealRecommendationError("boom")


def _build_request(
    *,
    day_start: str = "2026-04-20T09:00:00",
    day_end: str = "2026-04-20T11:00:00",
) -> DaySchedulingRequest:
    return DaySchedulingRequest.model_validate(
        {
            "day_start": day_start,
            "day_end": day_end,
            "food_budget_per_day": 40,
            "preferences": ["italian"],
            "acceptable_transport_modes": ["walking"],
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
                    "priority_score": 4,
                },
                {
                    "id": "colosseum",
                    "name": "Colosseum",
                    "location": {"latitude": 41.8902, "longitude": 12.4922},
                    "estimated_visit_duration_minutes": 90,
                    "estimated_cost": 18,
                    "category": "historical",
                    "rating": 4.9,
                    "summary": "Roman amphitheatre.",
                    "priority_score": 5,
                },
            ],
        }
    )


def test_planner_schedules_feasible_visits_by_priority() -> None:
    request = _build_request()

    response = PlannerService().plan_day(request)

    visit_events = [
        event for event in response.day_schedule.events if event.event_type == EVENT_TYPE_VISIT
    ]
    assert [event.place.id for event in visit_events] == ["colosseum"]
    assert visit_events[0].start_time.isoformat() == "2026-04-20T09:00:00"
    assert visit_events[0].end_time.isoformat() == "2026-04-20T10:30:00"
    assert [place.id for place in response.unscheduled_places] == ["pantheon"]
    assert response.warnings == [f"{WARNING_PLACE_UNSCHEDULED}:pantheon:insufficient_time"]


def test_planner_emits_travel_events_between_visits() -> None:
    request = _build_request(day_end="2026-04-20T12:00:00")

    response = PlannerService(FixedRouteClient(10)).plan_day(request)

    assert [event.event_type for event in response.day_schedule.events] == [
        EVENT_TYPE_VISIT,
        EVENT_TYPE_TRAVEL,
        EVENT_TYPE_VISIT,
    ]
    travel = response.day_schedule.events[1]
    assert travel.event_type == EVENT_TYPE_TRAVEL
    assert travel.estimated_travel_time_minutes == 10
    assert travel.transport_mode == "walking"


def test_planner_uses_fallback_when_route_estimation_fails() -> None:
    request = _build_request(day_end="2026-04-20T12:00:00")

    response = PlannerService(
        route_client=FailingRouteClient(),
        fallback_travel_minutes=7,
    ).plan_day(request)

    travel = response.day_schedule.events[1]
    assert travel.event_type == EVENT_TYPE_TRAVEL
    assert travel.estimated_travel_time_minutes == 7
    assert f"{WARNING_FALLBACK_TRAVEL}:pantheon:7" in response.warnings


def test_planner_applies_opening_hours_for_schedule_date() -> None:
    request = DaySchedulingRequest.model_validate(
        {
            "day_start": "2026-04-20T09:00:00",
            "day_end": "2026-04-20T12:00:00",
            "acceptable_transport_modes": ["walking"],
            "places": [
                {
                    "id": "pantheon",
                    "name": "Pantheon",
                    "location": {"latitude": 41.8986, "longitude": 12.4769},
                    "estimated_visit_duration_minutes": 45,
                    "priority_score": 5,
                    "opening_hours": [
                        {
                            "day_of_week": "monday",
                            "open_time": "10:00:00",
                            "close_time": "11:00:00",
                        }
                    ],
                }
            ],
        }
    )

    response = PlannerService().plan_day(request)

    visit = response.day_schedule.events[0]
    assert visit.event_type == EVENT_TYPE_VISIT
    assert visit.start_time.isoformat() == "2026-04-20T10:00:00"
    assert response.unscheduled_places == []


def test_planner_inserts_lunch_and_dinner_with_agent4_candidates() -> None:
    meal_client = CapturingMealClient()
    request = _build_request(day_end="2026-04-20T22:00:00")

    response = PlannerService(
        route_client=FixedRouteClient(10),
        meal_client=meal_client,
    ).plan_day(request)

    meal_events = [
        event for event in response.day_schedule.events if event.event_type == EVENT_TYPE_MEAL
    ]
    assert [event.meal_slot for event in meal_events] == ["lunch", "dinner"]
    assert all(event.restaurant is not None for event in meal_events)
    assert all(event.synthetic is False for event in meal_events)
    assert [request.time_of_day for request in meal_client.requests] == ["lunch", "dinner"]
    assert [request.budget_per_meal_per_person for request in meal_client.requests] == [
        20,
        20,
    ]


def test_planner_accounts_for_restaurant_meal_travel() -> None:
    request = _build_request(day_end="2026-04-20T15:00:00")
    request.places[1].estimated_visit_duration_minutes = 180

    response = PlannerService(
        route_client=FixedRouteClient(10),
        meal_client=CapturingMealClient(),
    ).plan_day(request)

    event_types = [event.event_type for event in response.day_schedule.events]
    assert event_types == [
        EVENT_TYPE_VISIT,
        EVENT_TYPE_TRAVEL,
        EVENT_TYPE_MEAL,
        EVENT_TYPE_TRAVEL,
        EVENT_TYPE_VISIT,
    ]
    assert response.day_schedule.events[1].estimated_travel_time_minutes == 10
    assert response.day_schedule.events[3].estimated_travel_time_minutes == 10


def test_planner_uses_synthetic_meal_when_agent4_fails() -> None:
    request = _build_request(day_end="2026-04-20T15:00:00")

    response = PlannerService(meal_client=FailingMealClient()).plan_day(request)

    meal = next(
        event
        for event in response.day_schedule.events
        if event.event_type == EVENT_TYPE_MEAL
    )
    assert meal.synthetic is True
    assert meal.restaurant is None
    assert f"{WARNING_AGENT4_UNAVAILABLE}:lunch" in response.warnings


def test_planner_uses_synthetic_meal_when_agent4_returns_no_candidates() -> None:
    request = _build_request(day_end="2026-04-20T15:00:00")

    response = PlannerService(meal_client=EmptyMealClient()).plan_day(request)

    meal = next(
        event
        for event in response.day_schedule.events
        if event.event_type == EVENT_TYPE_MEAL
    )
    assert meal.synthetic is True
    assert meal.restaurant is None
    assert f"{WARNING_NO_RESTAURANT_CANDIDATE}:lunch" in response.warnings


def test_planner_warns_when_meal_window_cannot_fit() -> None:
    request = _build_request(
        day_start="2026-04-20T13:30:00",
        day_end="2026-04-20T14:30:00",
    )

    response = PlannerService().plan_day(request)

    assert f"{WARNING_MEAL_NOT_INSERTED}:lunch:insufficient_time" in response.warnings
