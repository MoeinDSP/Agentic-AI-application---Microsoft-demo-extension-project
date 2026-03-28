from datetime import time
from functools import lru_cache

from agent3.models.plan import (
    DroppedPlace,
    PlannedStop,
    PlanRequest,
    PlanResponse,
)


class PlannerService:
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

        for place in prioritized_places:
            place_end_minutes = consumed_minutes + place.estimated_duration_minutes
            if place_end_minutes <= available_minutes:
                stop_start_minutes = self._time_to_minutes(request.day_start) + consumed_minutes
                stop_end_minutes = self._time_to_minutes(request.day_start) + place_end_minutes
                ordered_stops.append(
                    PlannedStop(
                        place_id=place.id,
                        place_name=place.name,
                        sequence=len(ordered_stops) + 1,
                        start_time=self._minutes_to_time(stop_start_minutes),
                        end_time=self._minutes_to_time(stop_end_minutes),
                        estimated_duration_minutes=place.estimated_duration_minutes,
                    )
                )
                consumed_minutes = place_end_minutes
                continue

            dropped_places.append(
                DroppedPlace(
                    place_id=place.id,
                    reason="insufficient_time",
                )
            )

        notes = [
            "deterministic_greedy_planner",
            "travel_time_assumption=zero",
            "feasibility=true_when_at_least_one_stop_is_scheduled",
            f"transport_preferences={','.join(request.transport_preferences) or 'none'}",
        ]

        return PlanResponse(
            ordered_stops=ordered_stops,
            dropped_places=dropped_places,
            notes=notes,
            feasibility=bool(ordered_stops),
        )

    def build_plan(self, request: PlanRequest) -> PlanResponse:
        return self.plan_day(request)

    def _time_to_minutes(self, value: time) -> int:
        return value.hour * 60 + value.minute

    def _minutes_to_time(self, total_minutes: int) -> time:
        hours, minutes = divmod(total_minutes, 60)
        return time(hour=hours, minute=minutes)


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    return PlannerService()
