from datetime import time
from functools import lru_cache
from typing import Protocol

from agent3.core.config import get_settings
from agent3.models.mcp import TravelEstimate
from agent3.models.plan import (
    Coordinates,
    DroppedPlace,
    PlaceInput,
    PlannedStop,
    PlanRequest,
    PlanResponse,
)
from agent3.services.mcp_client import RouteEstimationError, get_mcp_route_client


class RouteEstimator(Protocol):
    def estimate_route(
        self,
        *,
        origin: Coordinates,
        destination: Coordinates,
        transport_preferences: list[str],
    ) -> TravelEstimate: ...


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
        fallback_travel_minutes: int = 0,
    ) -> None:
        self._route_client = route_client or FixedTravelRouteClient(0)
        self._fallback_travel_minutes = fallback_travel_minutes

    def plan_day(self, request: PlanRequest) -> PlanResponse:
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

        for place in prioritized_places:
            travel_estimate = self._estimate_travel(
                origin=current_origin,
                destination=place,
                transport_preferences=request.transport_preferences,
            )
            if travel_estimate.source == "fallback":
                used_fallback = True
            arrival_offset_minutes = consumed_minutes + travel_estimate.estimated_duration_minutes
            place_end_minutes = arrival_offset_minutes + place.estimated_duration_minutes
            if place_end_minutes <= available_minutes:
                stop_arrival_minutes = (
                    self._time_to_minutes(request.day_start) + arrival_offset_minutes
                )
                stop_end_minutes = self._time_to_minutes(request.day_start) + place_end_minutes
                ordered_stops.append(
                    PlannedStop(
                        place_id=place.id,
                        place_name=place.name,
                        sequence=len(ordered_stops) + 1,
                        arrival_time=self._minutes_to_time(stop_arrival_minutes),
                        start_time=self._minutes_to_time(stop_arrival_minutes),
                        end_time=self._minutes_to_time(stop_end_minutes),
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
                    reason="insufficient_time",
                )
            )

        notes = [
            "deterministic_greedy_planner",
            "travel_time_source=mcp" if not used_fallback else "travel_time_source=fallback",
            "feasibility=true_when_at_least_one_stop_is_scheduled",
            f"transport_preferences={','.join(request.transport_preferences) or 'none'}",
        ]
        if used_fallback:
            notes.append(
                f"fallback_travel_minutes={self._fallback_travel_minutes}"
            )

        return PlanResponse(
            ordered_stops=ordered_stops,
            dropped_places=dropped_places,
            notes=notes,
            feasibility=bool(ordered_stops),
            total_travel_minutes=total_travel_minutes,
            total_visit_minutes=total_visit_minutes,
        )

    def build_plan(self, request: PlanRequest) -> PlanResponse:
        return self.plan_day(request)

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


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    settings = get_settings()
    return PlannerService(
        route_client=get_mcp_route_client(),
        fallback_travel_minutes=settings.fallback_travel_minutes,
    )
