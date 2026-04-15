"""
Final Itinerary Verification Service.

Validates the generated itinerary for:
  - Overlapping events
  - Realistic travel times between consecutive visits
  - Budget constraints respected
  - Visit durations are reasonable
"""
from __future__ import annotations

from datetime import timedelta
from math import atan2, cos, radians, sin, sqrt

from app.models import DailySchedule, MealEvent, VisitEvent


EARTH_RADIUS_KM = 6371.0
MAX_REALISTIC_SPEED_KMH = 60.0  # max urban travel speed (taxi/metro)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def verify_itinerary(
    daily_schedules: list[DailySchedule],
    total_budget: float | None = None,
    activity_budget: float | None = None,
    food_budget_total: float | None = None,
) -> list[str]:
    """
    Run all checks on the itinerary. Returns a list of warning strings.
    An empty list means everything passed.
    """
    warnings: list[str] = []

    for schedule in daily_schedules:
        day_label = schedule.date.isoformat()
        events = schedule.events

        if not events:
            warnings.append(f"Day {day_label}: no events scheduled.")
            continue

        # ── 1. Check for overlapping events ───────────────────────────────
        for i in range(len(events) - 1):
            curr = events[i]
            nxt  = events[i + 1]

            curr_end = _event_end_time(curr)
            nxt_start = _event_start_time(nxt)

            if curr_end is not None and nxt_start is not None:
                if curr_end > nxt_start:
                    overlap_min = int((curr_end - nxt_start).total_seconds() / 60)
                    warnings.append(
                        f"Day {day_label}: overlap of {overlap_min}min "
                        f"between event {i + 1} and {i + 2}."
                    )

        # ── 2. Check travel feasibility ───────────────────────────────────
        for i in range(len(events) - 1):
            curr = events[i]
            nxt  = events[i + 1]

            curr_loc = _event_location(curr)
            nxt_loc  = _event_location(nxt)
            curr_end = _event_end_time(curr)
            nxt_start = _event_start_time(nxt)

            if all(v is not None for v in [curr_loc, nxt_loc, curr_end, nxt_start]):
                dist = haversine_km(
                    curr_loc[0], curr_loc[1],
                    nxt_loc[0], nxt_loc[1],
                )
                gap_hours = (nxt_start - curr_end).total_seconds() / 3600

                if gap_hours > 0 and dist / gap_hours > MAX_REALISTIC_SPEED_KMH:
                    warnings.append(
                        f"Day {day_label}: travel between events {i + 1} and {i + 2} "
                        f"requires {dist:.1f}km in {gap_hours * 60:.0f}min — "
                        f"may be unrealistic."
                    )

        # ── 3. Check visit durations are reasonable ───────────────────────
        for i, event in enumerate(events):
            if isinstance(event, VisitEvent):
                dur = (event.end_time - event.start_time).total_seconds() / 60
                if dur < 10:
                    warnings.append(
                        f"Day {day_label}: event {i + 1} ({event.place.name}) "
                        f"has only {dur:.0f}min — too short?"
                    )
                if dur > 480:
                    warnings.append(
                        f"Day {day_label}: event {i + 1} ({event.place.name}) "
                        f"has {dur:.0f}min — unusually long."
                    )

    # ── 4. Budget check (aggregate) ───────────────────────────────────────
    if activity_budget is not None:
        total_activity_cost = 0.0
        for schedule in daily_schedules:
            for event in schedule.events:
                if isinstance(event, VisitEvent) and event.place.estimated_cost:
                    total_activity_cost += event.place.estimated_cost

        if total_activity_cost > activity_budget:
            warnings.append(
                f"Total activity cost ({total_activity_cost:.2f} EUR) "
                f"exceeds activity budget ({activity_budget:.2f} EUR)."
            )

    return warnings


# ── Internal helpers ──────────────────────────────────────────────────────────

def _event_start_time(event):
    if isinstance(event, VisitEvent):
        return event.start_time
    if isinstance(event, MealEvent):
        return event.time
    return None


def _event_end_time(event):
    if isinstance(event, VisitEvent):
        return event.end_time
    if isinstance(event, MealEvent):
        # Meals are assumed to take 60 minutes
        return event.time + timedelta(minutes=60)
    return None


def _event_location(event) -> tuple[float, float] | None:
    if isinstance(event, VisitEvent):
        return (event.place.location.latitude, event.place.location.longitude)
    if isinstance(event, MealEvent):
        loc = event.restaurant.location
        if isinstance(loc, dict):
            return (loc.get("latitude", 0.0), loc.get("longitude", 0.0))
        return (loc.latitude, loc.longitude)
    return None
