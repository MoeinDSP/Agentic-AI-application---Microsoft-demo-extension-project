from dataclasses import dataclass
from datetime import datetime, time, timedelta
from functools import lru_cache
from typing import Protocol

from agent3.core.config import get_settings
from agent3.models.agent4 import MealRecommendationRequest, RestaurantCandidate
from agent3.models.mcp import TravelEstimate
from agent3.models.plan import (
    DEFAULT_TRANSPORT_MODE,
    MEAL_SLOT_DINNER,
    MEAL_SLOT_LUNCH,
    WARNING_AGENT4_UNAVAILABLE,
    WARNING_FALLBACK_TRAVEL,
    WARNING_MEAL_NOT_INSERTED,
    WARNING_NO_RESTAURANT_CANDIDATE,
    WARNING_PLACE_UNSCHEDULED,
    Coordinates,
    DailySchedule,
    DaySchedulingRequest,
    DaySchedulingResult,
    Location,
    MealEvent,
    PlaceCandidate,
    RestaurantSummary,
    ScheduleEvent,
    TravelInfo,
    VisitEvent,
)
from agent3.services.agent4_client import MealRecommendationError, get_agent4_meal_client
from agent3.services.mcp_client import (
    RouteEstimationError,
    get_mcp_route_client,
)

MEAL_DURATION_MINUTES = 60
MEAL_WINDOWS = {
    MEAL_SLOT_LUNCH: (time(hour=12), time(hour=14)),
    MEAL_SLOT_DINNER: (time(hour=19), time(hour=21)),
}


class RouteEstimator(Protocol):
    def estimate_route(
        self,
        *,
        origin: Coordinates,
        destination: Coordinates,
        transport_preferences: list[str],
        departure_time: datetime | None = None,
    ) -> TravelEstimate: ...


class MealRecommender(Protocol):
    def recommend_meal(
        self,
        request: MealRecommendationRequest,
    ) -> object: ...


class FixedTravelRouteClient:
    def __init__(self, travel_minutes: int) -> None:
        self._travel_minutes = travel_minutes

    def estimate_route(
        self,
        *,
        origin: Coordinates,
        destination: Coordinates,
        transport_preferences: list[str],
        departure_time: datetime | None = None,
    ) -> TravelEstimate:
        _ = departure_time
        return TravelEstimate(
            source="fallback",
            mode=transport_preferences[0] if transport_preferences else DEFAULT_TRANSPORT_MODE,
            estimated_duration_minutes=self._travel_minutes,
            notes=[f"fixed_travel_minutes={self._travel_minutes}"],
        )


@dataclass(frozen=True)
class _TravelPlan:
    start_time: datetime
    end_time: datetime
    origin: Location
    destination: Location
    estimate: TravelEstimate


@dataclass(frozen=True)
class _MealInsertion:
    events: list[ScheduleEvent]
    current_time: datetime
    current_location: Location | None
    warnings: list[str]
    inserted: bool


class PlannerService:
    def __init__(
        self,
        route_client: RouteEstimator | None = None,
        meal_client: MealRecommender | None = None,
        fallback_travel_minutes: int = 0,
    ) -> None:
        self._route_client = route_client or FixedTravelRouteClient(0)
        self._meal_client = meal_client
        self._fallback_travel_minutes = fallback_travel_minutes

    def build_plan(self, request: DaySchedulingRequest) -> DaySchedulingResult:
        transport_mode = self._resolve_transport_mode(request.acceptable_transport_modes)
        meal_slots = self._meal_slots_for_day(request)
        per_meal_budget = self._per_meal_budget(request, meal_slots)
        pending_meals = set(meal_slots)
        events: list[ScheduleEvent] = []
        unscheduled_places: list[PlaceCandidate] = []
        warnings: list[str] = []
        current_time = request.day_start
        current_location: Location | None = None

        prioritized_places = sorted(
            enumerate(request.places),
            key=lambda indexed_place: (
                -(indexed_place[1].priority_score or 0),
                indexed_place[0],
            ),
        )

        for _, place in prioritized_places:
            current_time, current_location = self._insert_due_meals_before_place(
                request=request,
                place=place,
                transport_mode=transport_mode,
                pending_meals=pending_meals,
                per_meal_budget=per_meal_budget,
                current_time=current_time,
                current_location=current_location,
                events=events,
                warnings=warnings,
            )

            travel_plan = self._plan_travel(
                current_time=current_time,
                current_location=current_location,
                destination=place.location,
                transport_mode=transport_mode,
            )
            arrival_time = travel_plan.end_time if travel_plan is not None else current_time
            visit_start = self._apply_opening_hours(
                place=place,
                arrival_time=arrival_time,
                schedule_date=request.day_start,
            )
            visit_end = visit_start + timedelta(
                minutes=place.estimated_visit_duration_minutes
            )
            drop_reason = self._get_drop_reason(
                request=request,
                place=place,
                visit_start=visit_start,
                visit_end=visit_end,
            )
            if drop_reason is not None:
                unscheduled_places.append(place)
                warnings.append(f"{WARNING_PLACE_UNSCHEDULED}:{place.id}:{drop_reason}")
                continue

            if travel_plan is not None:
                events.append(self._build_travel_event(travel_plan, transport_mode))
                if travel_plan.estimate.source == "fallback":
                    warnings.append(
                        f"{WARNING_FALLBACK_TRAVEL}:{place.id}:"
                        f"{self._fallback_travel_minutes}"
                    )

            events.append(
                VisitEvent(
                    start_time=visit_start,
                    end_time=visit_end,
                    place=place,
                )
            )
            current_time = visit_end
            current_location = place.location

        for meal_slot in list(meal_slots):
            if meal_slot not in pending_meals:
                continue
            next_location = request.places[0].location if request.places else None
            insertion = self._try_insert_meal(
                request=request,
                meal_slot=meal_slot,
                current_time=current_time,
                current_location=current_location,
                next_location=next_location,
                transport_mode=transport_mode,
                per_meal_budget=per_meal_budget,
            )
            pending_meals.remove(meal_slot)
            events.extend(insertion.events)
            warnings.extend(insertion.warnings)
            if insertion.inserted:
                current_time = insertion.current_time
                current_location = insertion.current_location

        return DaySchedulingResult(
            day_schedule=DailySchedule(date=request.day_start.date(), events=events),
            unscheduled_places=unscheduled_places,
            warnings=warnings,
        )

    def plan_day(self, request: DaySchedulingRequest) -> DaySchedulingResult:
        return self.build_plan(request)

    def _insert_due_meals_before_place(
        self,
        *,
        request: DaySchedulingRequest,
        place: PlaceCandidate,
        transport_mode: str,
        pending_meals: set[str],
        per_meal_budget: float | None,
        current_time: datetime,
        current_location: Location | None,
        events: list[ScheduleEvent],
        warnings: list[str],
    ) -> tuple[datetime, Location | None]:
        while True:
            due_slot = self._next_due_meal_slot(
                request=request,
                pending_meals=pending_meals,
                current_time=current_time,
                current_location=current_location,
                next_place=place,
                transport_mode=transport_mode,
            )
            if due_slot is None:
                return current_time, current_location

            insertion = self._try_insert_meal(
                request=request,
                meal_slot=due_slot,
                current_time=current_time,
                current_location=current_location,
                next_location=place.location,
                transport_mode=transport_mode,
                per_meal_budget=per_meal_budget,
            )
            pending_meals.remove(due_slot)
            events.extend(insertion.events)
            warnings.extend(insertion.warnings)
            if insertion.inserted:
                current_time = insertion.current_time
                current_location = insertion.current_location

    def _next_due_meal_slot(
        self,
        *,
        request: DaySchedulingRequest,
        pending_meals: set[str],
        current_time: datetime,
        current_location: Location | None,
        next_place: PlaceCandidate,
        transport_mode: str,
    ) -> str | None:
        for meal_slot in sorted(
            pending_meals,
            key=lambda slot: self._meal_window_datetimes(request, slot)[0],
        ):
            window_start, window_end = self._meal_window_datetimes(request, meal_slot)
            travel_plan = self._plan_travel(
                current_time=current_time,
                current_location=current_location,
                destination=next_place.location,
                transport_mode=transport_mode,
            )
            arrival_time = travel_plan.end_time if travel_plan is not None else current_time
            projected_visit_end = arrival_time + timedelta(
                minutes=next_place.estimated_visit_duration_minutes
            )
            if current_time >= window_start or projected_visit_end > window_start:
                if current_time <= window_end:
                    return meal_slot
        return None

    def _try_insert_meal(
        self,
        *,
        request: DaySchedulingRequest,
        meal_slot: str,
        current_time: datetime,
        current_location: Location | None,
        next_location: Location | None,
        transport_mode: str,
        per_meal_budget: float | None,
    ) -> _MealInsertion:
        window_start, window_end = self._meal_window_datetimes(request, meal_slot)
        meal_start = max(current_time, window_start)
        meal_end = meal_start + timedelta(minutes=MEAL_DURATION_MINUTES)
        if meal_end > window_end or meal_end > request.day_end:
            return _MealInsertion(
                events=[],
                current_time=current_time,
                current_location=current_location,
                warnings=[f"{WARNING_MEAL_NOT_INSERTED}:{meal_slot}:insufficient_time"],
                inserted=False,
            )

        search_location = current_location or next_location
        if search_location is None:
            return self._synthetic_meal_insertion(
                meal_slot=meal_slot,
                meal_start=meal_start,
                meal_end=meal_end,
                current_location=current_location,
                warning=f"{WARNING_NO_RESTAURANT_CANDIDATE}:{meal_slot}",
            )

        candidate, warning = self._recommend_restaurant(
            meal_slot=meal_slot,
            search_location=search_location,
            request=request,
            per_meal_budget=per_meal_budget,
        )
        if candidate is None:
            return self._synthetic_meal_insertion(
                meal_slot=meal_slot,
                meal_start=meal_start,
                meal_end=meal_end,
                current_location=current_location or search_location,
                warning=warning or f"{WARNING_NO_RESTAURANT_CANDIDATE}:{meal_slot}",
            )

        restaurant = self._restaurant_summary(candidate)
        travel_plan = self._plan_travel(
            current_time=current_time,
            current_location=current_location,
            destination=restaurant.location,
            transport_mode=transport_mode,
        )
        events: list[ScheduleEvent] = []
        warnings: list[str] = []
        if travel_plan is not None:
            arrival_time = travel_plan.end_time
            restaurant_meal_start = max(arrival_time, window_start)
            restaurant_meal_end = restaurant_meal_start + timedelta(
                minutes=MEAL_DURATION_MINUTES
            )
            if restaurant_meal_end > window_end or restaurant_meal_end > request.day_end:
                return self._synthetic_meal_insertion(
                    meal_slot=meal_slot,
                    meal_start=meal_start,
                    meal_end=meal_end,
                    current_location=current_location,
                    warning=f"{WARNING_MEAL_NOT_INSERTED}:{meal_slot}:restaurant_travel_time",
                )
            events.append(self._build_travel_event(travel_plan, transport_mode))
            if travel_plan.estimate.source == "fallback":
                warnings.append(
                    f"{WARNING_FALLBACK_TRAVEL}:{meal_slot}:"
                    f"{self._fallback_travel_minutes}"
                )
            meal_start = restaurant_meal_start
            meal_end = restaurant_meal_end

        events.append(
            MealEvent(
                meal_slot=meal_slot,  # type: ignore[arg-type]
                start_time=meal_start,
                end_time=meal_end,
                restaurant=restaurant,
                synthetic=False,
            )
        )
        return _MealInsertion(
            events=events,
            current_time=meal_end,
            current_location=restaurant.location,
            warnings=warnings,
            inserted=True,
        )

    def _synthetic_meal_insertion(
        self,
        *,
        meal_slot: str,
        meal_start: datetime,
        meal_end: datetime,
        current_location: Location | None,
        warning: str,
    ) -> _MealInsertion:
        return _MealInsertion(
            events=[
                MealEvent(
                    meal_slot=meal_slot,  # type: ignore[arg-type]
                    start_time=meal_start,
                    end_time=meal_end,
                    restaurant=None,
                    synthetic=True,
                )
            ],
            current_time=meal_end,
            current_location=current_location,
            warnings=[warning],
            inserted=True,
        )

    def _recommend_restaurant(
        self,
        *,
        meal_slot: str,
        search_location: Location,
        request: DaySchedulingRequest,
        per_meal_budget: float | None,
    ) -> tuple[RestaurantCandidate | None, str | None]:
        if self._meal_client is None:
            return None, f"{WARNING_AGENT4_UNAVAILABLE}:{meal_slot}"

        meal_request = MealRecommendationRequest(
            time_of_day=meal_slot,  # type: ignore[arg-type]
            search_center=self._to_coordinates(search_location),
            search_radius_meters=2500,
            budget_per_meal_per_person=per_meal_budget,
            preferences=request.preferences,
        )

        try:
            response = self._meal_client.recommend_meal(meal_request)
        except MealRecommendationError:
            return None, f"{WARNING_AGENT4_UNAVAILABLE}:{meal_slot}"

        candidates = getattr(response, "candidates", [])
        if not candidates:
            return None, f"{WARNING_NO_RESTAURANT_CANDIDATE}:{meal_slot}"
        return candidates[0], None

    def _plan_travel(
        self,
        *,
        current_time: datetime,
        current_location: Location | None,
        destination: Location,
        transport_mode: str,
    ) -> _TravelPlan | None:
        if current_location is None:
            return None
        estimate = self._estimate_travel(
            origin=current_location,
            destination=destination,
            transport_preferences=[transport_mode],
            departure_time=current_time,
        )
        return _TravelPlan(
            start_time=current_time,
            end_time=current_time
            + timedelta(minutes=estimate.estimated_duration_minutes),
            origin=current_location,
            destination=destination,
            estimate=estimate,
        )

    def _estimate_travel(
        self,
        *,
        origin: Location,
        destination: Location,
        transport_preferences: list[str],
        departure_time: datetime | None = None,
    ) -> TravelEstimate:
        try:
            return self._route_client.estimate_route(
                origin=self._to_coordinates(origin),
                destination=self._to_coordinates(destination),
                transport_preferences=transport_preferences,
                departure_time=departure_time,
            )
        except RouteEstimationError:
            return TravelEstimate(
                source="fallback",
                mode=transport_preferences[0] if transport_preferences else DEFAULT_TRANSPORT_MODE,
                estimated_duration_minutes=self._fallback_travel_minutes,
                notes=[f"fallback_travel_minutes={self._fallback_travel_minutes}"],
            )

    def _build_travel_event(
        self,
        travel_plan: _TravelPlan,
        transport_mode: str,
    ) -> TravelInfo:
        return TravelInfo(
            start_time=travel_plan.start_time,
            end_time=travel_plan.end_time,
            origin=travel_plan.origin,
            destination=travel_plan.destination,
            transport_mode=transport_mode,
            transport_description=(
                f"{transport_mode} route from "
                f"{travel_plan.origin.latitude},{travel_plan.origin.longitude} to "
                f"{travel_plan.destination.latitude},{travel_plan.destination.longitude}"
            ),
            estimated_travel_time_minutes=travel_plan.estimate.estimated_duration_minutes,
        )

    def _apply_opening_hours(
        self,
        *,
        place: PlaceCandidate,
        arrival_time: datetime,
        schedule_date: datetime,
    ) -> datetime:
        opening_hours = self._opening_hours_for_date(place, schedule_date)
        if opening_hours is None:
            return arrival_time
        open_dt = datetime.combine(arrival_time.date(), opening_hours.open_time)
        return max(arrival_time, open_dt)

    def _get_drop_reason(
        self,
        *,
        request: DaySchedulingRequest,
        place: PlaceCandidate,
        visit_start: datetime,
        visit_end: datetime,
    ) -> str | None:
        if visit_end > request.day_end:
            return "insufficient_time"

        opening_hours = self._opening_hours_for_date(place, request.day_start)
        if opening_hours is None:
            return None

        close_dt = datetime.combine(visit_start.date(), opening_hours.close_time)
        if visit_start >= close_dt:
            return "closed_at_arrival"
        if visit_end > close_dt:
            return "closes_before_visit_ends"
        return None

    def _opening_hours_for_date(
        self,
        place: PlaceCandidate,
        schedule_date: datetime,
    ):
        day_name = schedule_date.strftime("%A").lower()
        for entry in place.opening_hours:
            if entry.day_of_week == day_name:
                return entry
        return None

    def _meal_slots_for_day(self, request: DaySchedulingRequest) -> list[str]:
        return [
            meal_slot
            for meal_slot in (MEAL_SLOT_LUNCH, MEAL_SLOT_DINNER)
            if self._window_overlaps_day(request, meal_slot)
        ]

    def _window_overlaps_day(self, request: DaySchedulingRequest, meal_slot: str) -> bool:
        window_start, window_end = self._meal_window_datetimes(request, meal_slot)
        return request.day_start < window_end and request.day_end > window_start

    def _meal_window_datetimes(
        self,
        request: DaySchedulingRequest,
        meal_slot: str,
    ) -> tuple[datetime, datetime]:
        window_start, window_end = MEAL_WINDOWS[meal_slot]
        schedule_date = request.day_start.date()
        return (
            datetime.combine(schedule_date, window_start),
            datetime.combine(schedule_date, window_end),
        )

    def _per_meal_budget(
        self,
        request: DaySchedulingRequest,
        meal_slots: list[str],
    ) -> float | None:
        if request.food_budget_per_day is None or not meal_slots:
            return None
        return request.food_budget_per_day / len(meal_slots)

    def _resolve_transport_mode(self, transport_modes: list[str]) -> str:
        if transport_modes:
            return transport_modes[0]
        return get_settings().default_transport_mode

    def _to_coordinates(self, location: Location) -> Coordinates:
        return Coordinates(lat=location.latitude, lng=location.longitude)

    def _restaurant_summary(self, candidate: RestaurantCandidate) -> RestaurantSummary:
        return RestaurantSummary(
            id=candidate.id,
            name=candidate.name,
            location=Location(
                latitude=candidate.location.lat,
                longitude=candidate.location.lng,
            ),
            price_level=candidate.price_level,
            cuisines=candidate.cuisines,
            rating=candidate.rating,
            summary=candidate.summary,
        )


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    settings = get_settings()
    return PlannerService(
        route_client=get_mcp_route_client(),
        meal_client=get_agent4_meal_client(),
        fallback_travel_minutes=settings.fallback_travel_minutes,
    )
