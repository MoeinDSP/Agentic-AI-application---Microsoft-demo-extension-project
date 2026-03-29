from datetime import time
from functools import lru_cache
from typing import Protocol

from agent3.core.config import get_settings
from agent3.models.agent4 import MealRecommendationRequest, RestaurantCandidate
from agent3.models.mcp import TravelEstimate
from agent3.models.plan import (
    AGENT4_UNAVAILABLE_USING_SYNTHETIC_LUNCH_NOTE,
    DROP_REASON_CLOSED_AT_ARRIVAL,
    DROP_REASON_CLOSES_BEFORE_VISIT_ENDS,
    DROP_REASON_INSUFFICIENT_TIME,
    LUNCH_INSERTED_NOTE,
    LUNCH_NOT_INSERTED_NOTE,
    NO_RESTAURANT_CANDIDATE_FOUND_NOTE,
    STOP_TYPE_LUNCH,
    STOP_TYPE_MEAL,
    Coordinates,
    DroppedPlace,
    PlaceInput,
    PlannedStop,
    PlanRequest,
    PlanResponse,
    RestaurantSummary,
)
from agent3.services.agent4_client import MealRecommendationError, get_agent4_meal_client
from agent3.services.mcp_client import (
    RouteEstimationError,
    get_mcp_route_client,
)


class RouteEstimator(Protocol):
    def estimate_route(
        self,
        *,
        origin: Coordinates,
        destination: Coordinates,
        transport_preferences: list[str],
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
    ) -> TravelEstimate:
        return TravelEstimate(
            source="fallback",
            mode=transport_preferences[0] if transport_preferences else "walk",
            estimated_duration_minutes=self._travel_minutes,
            notes=[f"fixed_travel_minutes={self._travel_minutes}"],
        )


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

    def build_plan(self, request: PlanRequest) -> PlanResponse:
        selected_transport_mode = self._resolve_transport_mode(request.transport_preferences)
        available_minutes = self._time_to_minutes(request.day_end) - self._time_to_minutes(
            request.day_start
        )
        prioritized_places = sorted(
            request.places,
            key=lambda place: -place.priority,
        )

        ordered_stops: list[PlannedStop] = []
        dropped_places: list[DroppedPlace] = []
        consumed_minutes = 0
        total_travel_minutes = 0
        total_visit_minutes = 0
        current_origin = request.start_location
        used_fallback = False
        lunch_inserted = False
        agent4_unavailable = False
        no_restaurant_candidate_found = False

        for place in prioritized_places:
            travel_estimate = self._estimate_travel(
                origin=current_origin,
                destination=place,
                transport_preferences=[selected_transport_mode],
            )
            if travel_estimate.source == "fallback":
                used_fallback = True

            lunch_stop = self._build_lunch_stop_before_place(
                request=request,
                consumed_minutes=consumed_minutes,
                projected_place_end_minutes=(
                    self._time_to_minutes(request.day_start)
                    + consumed_minutes
                    + travel_estimate.estimated_duration_minutes
                    + place.estimated_duration_minutes
                ),
                sequence=len(ordered_stops) + 1,
                lunch_inserted=lunch_inserted,
            )
            if lunch_stop is not None:
                lunch_stop, unavailable, no_candidates = self._maybe_enrich_lunch_stop(
                    lunch_stop=lunch_stop,
                    current_origin=current_origin,
                    request=request,
                )
                ordered_stops.append(lunch_stop)
                lunch_inserted = True
                agent4_unavailable = agent4_unavailable or unavailable
                no_restaurant_candidate_found = (
                    no_restaurant_candidate_found or no_candidates
                )
                consumed_minutes = (
                    self._time_to_minutes(lunch_stop.end_time)
                    - self._time_to_minutes(request.day_start)
                )
                total_visit_minutes += request.lunch_duration_minutes

            arrival_offset_minutes = consumed_minutes + travel_estimate.estimated_duration_minutes
            place_end_minutes = arrival_offset_minutes + place.estimated_duration_minutes
            stop_arrival_minutes = (
                self._time_to_minutes(request.day_start) + arrival_offset_minutes
            )
            stop_end_minutes = self._time_to_minutes(request.day_start) + place_end_minutes
            arrival_time = self._minutes_to_time(stop_arrival_minutes)
            end_time = self._minutes_to_time(stop_end_minutes)

            opening_hours_drop_reason = self._get_opening_hours_drop_reason(
                place=place,
                arrival_time=arrival_time,
                end_time=end_time,
            )
            if opening_hours_drop_reason is not None:
                dropped_places.append(
                    DroppedPlace(
                        place_id=place.id,
                        reason=opening_hours_drop_reason,
                    )
                )
                continue

            if place_end_minutes <= available_minutes:
                ordered_stops.append(
                    PlannedStop(
                        place_id=place.id,
                        place_name=place.name,
                        sequence=len(ordered_stops) + 1,
                        arrival_time=arrival_time,
                        start_time=arrival_time,
                        end_time=end_time,
                        travel_minutes_from_previous=travel_estimate.estimated_duration_minutes,
                        estimated_duration_minutes=place.estimated_duration_minutes,
                    )
                )
                consumed_minutes = place_end_minutes
                total_travel_minutes += travel_estimate.estimated_duration_minutes
                total_visit_minutes += place.estimated_duration_minutes
                current_origin = Coordinates(lat=place.lat, lng=place.lng)
                continue

            dropped_places.append(
                DroppedPlace(
                    place_id=place.id,
                    reason=DROP_REASON_INSUFFICIENT_TIME,
                )
            )

        lunch_stop = self._build_lunch_stop_after_places(
            request=request,
            consumed_minutes=consumed_minutes,
            sequence=len(ordered_stops) + 1,
            lunch_inserted=lunch_inserted,
        )
        if lunch_stop is not None:
            lunch_stop, unavailable, no_candidates = self._maybe_enrich_lunch_stop(
                lunch_stop=lunch_stop,
                current_origin=current_origin,
                request=request,
            )
            ordered_stops.append(lunch_stop)
            lunch_inserted = True
            agent4_unavailable = agent4_unavailable or unavailable
            no_restaurant_candidate_found = no_restaurant_candidate_found or no_candidates
            total_visit_minutes += request.lunch_duration_minutes

        notes = [
            "deterministic_greedy_planner",
            "travel_time_source=mcp" if not used_fallback else "travel_time_source=fallback",
            "feasibility=true_when_at_least_one_stop_is_scheduled",
            f"selected_transport_mode={selected_transport_mode}",
            f"transport_preferences={','.join(request.transport_preferences) or 'none'}",
        ]
        if request.lunch_required:
            notes.append(
                LUNCH_INSERTED_NOTE if lunch_inserted else LUNCH_NOT_INSERTED_NOTE
            )
            if self._meal_client is not None:
                notes.append(
                    f"agent4_invocation_mode={get_settings().agent4_invocation_mode}"
                )
        if agent4_unavailable:
            notes.append(AGENT4_UNAVAILABLE_USING_SYNTHETIC_LUNCH_NOTE)
        if no_restaurant_candidate_found:
            notes.append(NO_RESTAURANT_CANDIDATE_FOUND_NOTE)
        if used_fallback:
            notes.append(
                f"fallback_travel_minutes={self._fallback_travel_minutes}"
            )

        return PlanResponse(
            ordered_stops=ordered_stops,
            dropped_places=dropped_places,
            notes=notes,
            feasibility=bool(ordered_stops),
            selected_transport_mode=selected_transport_mode,
            total_travel_minutes=total_travel_minutes,
            total_visit_minutes=total_visit_minutes,
        )

    def plan_day(self, request: PlanRequest) -> PlanResponse:
        return self.build_plan(request)

    def _estimate_travel(
        self,
        *,
        origin: Coordinates,
        destination: PlaceInput,
        transport_preferences: list[str],
    ) -> TravelEstimate:
        try:
            return self._route_client.estimate_route(
                origin=origin,
                destination=Coordinates(lat=destination.lat, lng=destination.lng),
                transport_preferences=transport_preferences,
            )
        except RouteEstimationError:
            return TravelEstimate(
                source="fallback",
                mode=transport_preferences[0] if transport_preferences else "walk",
                estimated_duration_minutes=self._fallback_travel_minutes,
                notes=[f"fallback_travel_minutes={self._fallback_travel_minutes}"],
            )

    def _time_to_minutes(self, value: time) -> int:
        return value.hour * 60 + value.minute

    def _minutes_to_time(self, total_minutes: int) -> time:
        hours, minutes = divmod(total_minutes, 60)
        return time(hour=hours, minute=minutes)

    def _resolve_transport_mode(self, transport_preferences: list[str]) -> str:
        if transport_preferences:
            return transport_preferences[0]
        return get_settings().default_transport_mode

    def _maybe_enrich_lunch_stop(
        self,
        *,
        lunch_stop: PlannedStop,
        current_origin: Coordinates,
        request: PlanRequest,
    ) -> tuple[PlannedStop, bool, bool]:
        if self._meal_client is None:
            return lunch_stop, False, False

        meal_request = MealRecommendationRequest(
            time_of_day="lunch",
            search_center=current_origin,
            search_radius_meters=2500,
            budget_per_meal_per_person=request.budget_per_meal_per_person,
            preferences=request.meal_preferences,
        )

        try:
            response = self._meal_client.recommend_meal(meal_request)
        except MealRecommendationError:
            return lunch_stop, True, False

        candidates = getattr(response, "candidates", [])
        if not candidates:
            return lunch_stop, False, True

        return self._apply_restaurant_to_lunch_stop(
            lunch_stop=lunch_stop,
            candidate=candidates[0],
        ), False, False

    def _apply_restaurant_to_lunch_stop(
        self,
        *,
        lunch_stop: PlannedStop,
        candidate: RestaurantCandidate,
    ) -> PlannedStop:
        return lunch_stop.model_copy(
            update={
                "stop_type": STOP_TYPE_MEAL,
                "place_id": candidate.id,
                "place_name": candidate.name,
                "restaurant": RestaurantSummary.model_validate(candidate.model_dump()),
            }
        )

    def _build_lunch_stop_before_place(
        self,
        *,
        request: PlanRequest,
        consumed_minutes: int,
        projected_place_end_minutes: int,
        sequence: int,
        lunch_inserted: bool,
    ) -> PlannedStop | None:
        if lunch_inserted or not request.lunch_required:
            return None

        current_time_minutes = self._time_to_minutes(request.day_start) + consumed_minutes
        lunch_window_start = self._time_to_minutes(request.lunch_time_window_start)
        lunch_window_end = self._time_to_minutes(request.lunch_time_window_end)
        lunch_duration = request.lunch_duration_minutes
        latest_lunch_start = lunch_window_end - lunch_duration

        should_insert_now = (
            current_time_minutes >= lunch_window_start
            or current_time_minutes > latest_lunch_start
            or projected_place_end_minutes > lunch_window_end
        )
        if not should_insert_now:
            return None

        return self._build_lunch_stop(
            request=request,
            current_time_minutes=current_time_minutes,
            sequence=sequence,
        )

    def _build_lunch_stop_after_places(
        self,
        *,
        request: PlanRequest,
        consumed_minutes: int,
        sequence: int,
        lunch_inserted: bool,
    ) -> PlannedStop | None:
        if lunch_inserted or not request.lunch_required:
            return None

        current_time_minutes = self._time_to_minutes(request.day_start) + consumed_minutes
        return self._build_lunch_stop(
            request=request,
            current_time_minutes=current_time_minutes,
            sequence=sequence,
        )

    def _build_lunch_stop(
        self,
        *,
        request: PlanRequest,
        current_time_minutes: int,
        sequence: int,
    ) -> PlannedStop | None:
        lunch_window_start = self._time_to_minutes(request.lunch_time_window_start)
        lunch_window_end = self._time_to_minutes(request.lunch_time_window_end)
        lunch_duration = request.lunch_duration_minutes
        lunch_start_minutes = max(current_time_minutes, lunch_window_start)
        lunch_end_minutes = lunch_start_minutes + lunch_duration

        if lunch_end_minutes > lunch_window_end:
            return None
        if lunch_end_minutes > self._time_to_minutes(request.day_end):
            return None

        lunch_start_time = self._minutes_to_time(lunch_start_minutes)
        lunch_end_time = self._minutes_to_time(lunch_end_minutes)
        return PlannedStop(
            stop_type=STOP_TYPE_LUNCH,
            place_id="lunch",
            place_name="Lunch",
            sequence=sequence,
            arrival_time=lunch_start_time,
            start_time=lunch_start_time,
            end_time=lunch_end_time,
            travel_minutes_from_previous=0,
            estimated_duration_minutes=lunch_duration,
        )

    def _get_opening_hours_drop_reason(
        self,
        *,
        place: PlaceInput,
        arrival_time: time,
        end_time: time,
    ) -> str | None:
        if place.opens_at is None and place.closes_at is None:
            return None

        if place.opens_at is not None and arrival_time < place.opens_at:
            return DROP_REASON_CLOSED_AT_ARRIVAL

        if place.closes_at is not None and arrival_time >= place.closes_at:
            return DROP_REASON_CLOSED_AT_ARRIVAL

        if place.closes_at is not None and end_time > place.closes_at:
            return DROP_REASON_CLOSES_BEFORE_VISIT_ENDS

        return None


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    settings = get_settings()
    return PlannerService(
        route_client=get_mcp_route_client(),
        meal_client=get_agent4_meal_client(),
        fallback_travel_minutes=settings.fallback_travel_minutes,
    )
