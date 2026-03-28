from datetime import datetime, timedelta
from functools import lru_cache

from agent3.models.plan import (
    DroppedPlace,
    PlannedStop,
    PlanRequest,
    PlanResponse,
)


class PlannerService:
    def build_plan(self, request: PlanRequest) -> PlanResponse:
        available_minutes = (
            request.day_end.hour * 60
            + request.day_end.minute
            - request.day_start.hour * 60
            - request.day_start.minute
        )

        prioritized_places = sorted(
            request.places,
            key=lambda place: (-place.priority, place.estimated_duration_minutes, place.id),
        )

        ordered_stops: list[PlannedStop] = []
        dropped_places: list[DroppedPlace] = []
        consumed_minutes = 0

        for sequence, place in enumerate(prioritized_places, start=1):
            if consumed_minutes + place.estimated_duration_minutes <= available_minutes:
                stop_start = self._offset_time(request.day_start, consumed_minutes)
                stop_end = self._offset_time(
                    request.day_start,
                    consumed_minutes + place.estimated_duration_minutes,
                )
                ordered_stops.append(
                    PlannedStop(
                        place_id=place.id,
                        place_name=place.name,
                        sequence=sequence,
                        start_time=stop_start,
                        end_time=stop_end,
                        estimated_duration_minutes=place.estimated_duration_minutes,
                    )
                )
                consumed_minutes += place.estimated_duration_minutes
                continue

            dropped_places.append(
                DroppedPlace(
                    place_id=place.id,
                    reason="outside_placeholder_day_capacity",
                )
            )

        notes = [
            "placeholder_plan_generated",
            f"transport_preferences={','.join(request.transport_preferences) or 'none'}",
        ]

        return PlanResponse(
            ordered_stops=ordered_stops,
            dropped_places=dropped_places,
            notes=notes,
            feasibility=not dropped_places,
        )

    def _offset_time(self, base_time, offset_minutes: int):
        base_datetime = datetime.combine(datetime.today(), base_time)
        return (base_datetime + timedelta(minutes=offset_minutes)).time()


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    return PlannerService()
