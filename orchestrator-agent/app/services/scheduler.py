"""
Agent 3 — Single Day Scheduler (embedded in orchestrator).

Creates a chronological daily plan from a cluster of places:
  - Orders places to minimise back-and-forth travel
  - Allocates realistic visit durations
  - Inserts meal breaks (lunch + dinner) at appropriate times
  - Calls Agent 4 for restaurant recommendations when a meal is needed
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from math import atan2, cos, radians, sin, sqrt
from typing import Optional

from app.models import (
    DailySchedule,
    MealEvent,
    MealSlot,
    PlaceCandidate,
    RestaurantCandidate,
    VisitEvent,
)
from app.services.a2a_client_helper import call_food_recommender


# ── Constants ─────────────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

# Typical walking speed in a city (~4.5 km/h → ~13 min/km)
TRAVEL_MINUTES_PER_KM = 13.0

# Meal time windows (24h clock)
LUNCH_WINDOW  = (12, 14)   # lunch between 12:00–14:00
DINNER_WINDOW = (19, 21)   # dinner between 19:00–21:00

# Default meal duration in minutes
MEAL_DURATION_MINUTES = 60

# Default search radius for restaurants (meters)
RESTAURANT_SEARCH_RADIUS = 800


# ── Geo helpers ───────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def estimate_travel_minutes(place_a: PlaceCandidate, place_b: PlaceCandidate) -> int:
    dist = haversine_km(
        place_a.location.latitude, place_a.location.longitude,
        place_b.location.latitude, place_b.location.longitude,
    )
    return max(5, int(dist * TRAVEL_MINUTES_PER_KM))


# ── Greedy nearest-neighbour ordering ────────────────────────────────────────

def order_places_greedy(places: list[PlaceCandidate]) -> list[PlaceCandidate]:
    """
    Reorder places using a greedy nearest-neighbour heuristic
    to reduce total travel distance within the day.
    """
    if len(places) <= 2:
        return list(places)

    remaining = list(places)

    # Start with the highest-priority place
    remaining.sort(key=lambda p: -(p.priority_score or 0.0))
    ordered = [remaining.pop(0)]

    while remaining:
        last = ordered[-1]
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: haversine_km(
                last.location.latitude, last.location.longitude,
                remaining[i].location.latitude, remaining[i].location.longitude,
            ),
        )
        ordered.append(remaining.pop(nearest_idx))

    return ordered


# ── Meal insertion logic ──────────────────────────────────────────────────────

def _needs_meal(current_time: datetime, meals_inserted: set[str]) -> Optional[MealSlot]:
    """Check if a meal should be inserted at the current time."""
    hour = current_time.hour

    if "lunch" not in meals_inserted and LUNCH_WINDOW[0] <= hour < LUNCH_WINDOW[1]:
        return MealSlot.lunch

    if "dinner" not in meals_inserted and DINNER_WINDOW[0] <= hour < DINNER_WINDOW[1]:
        return MealSlot.dinner

    return None


def _centroid(places: list[PlaceCandidate]) -> tuple[float, float]:
    """Average lat/lon of a list of places."""
    if not places:
        return (0.0, 0.0)
    lat = sum(p.location.latitude for p in places) / len(places)
    lon = sum(p.location.longitude for p in places) / len(places)
    return lat, lon


# ── Main scheduler ────────────────────────────────────────────────────────────

async def schedule_day(
    places: list[PlaceCandidate],
    day_date: date,
    day_start_hour: int = 9,
    day_end_hour: int = 22,
    food_budget_per_day: float | None = None,
    preferences: list[str] | None = None,
) -> DailySchedule:
    """
    Build a DailySchedule for one day from a cluster of places.

    Pipeline:
      1. Order places by proximity (greedy nearest-neighbour)
      2. Walk through the ordered list, assigning start/end times
      3. When current_time enters a meal window, call Agent 4 and insert a MealEvent
      4. Skip remaining places if we run past day_end
    """
    ordered = order_places_greedy(places)

    current_time = datetime(day_date.year, day_date.month, day_date.day, day_start_hour, 0)
    day_end = datetime(day_date.year, day_date.month, day_date.day, day_end_hour, 0)

    events: list[VisitEvent | MealEvent] = []
    meals_inserted: set[str] = set()
    place_idx = 0

    # Budget per meal: split daily food budget across 2 meals (lunch + dinner)
    budget_per_meal = None
    if food_budget_per_day is not None:
        budget_per_meal = food_budget_per_day / 2.0

    while place_idx < len(ordered) and current_time < day_end:

        # ── Check if we need a meal first ─────────────────────────────────
        meal_slot = _needs_meal(current_time, meals_inserted)

        if meal_slot is not None:
            # Use the last visited place as search center, or cluster centroid
            if events:
                last_event = events[-1]
                if isinstance(last_event, VisitEvent):
                    search_lat = last_event.place.location.latitude
                    search_lon = last_event.place.location.longitude
                else:
                    search_lat = last_event.restaurant.location.latitude
                    search_lon = last_event.restaurant.location.longitude
            else:
                search_lat, search_lon = _centroid(ordered)

            restaurant = await _find_restaurant(
                meal_slot=meal_slot.value,
                latitude=search_lat,
                longitude=search_lon,
                budget_per_person=budget_per_meal,
                preferences=preferences,
            )

            if restaurant:
                events.append(MealEvent(
                    time=current_time,
                    meal_slot=meal_slot,
                    restaurant=restaurant,
                ))

            meals_inserted.add(meal_slot.value)
            current_time += timedelta(minutes=MEAL_DURATION_MINUTES)

            # Add travel time from restaurant to next place
            if place_idx < len(ordered) and restaurant:
                travel = max(5, int(
                    haversine_km(
                        restaurant.location.latitude,
                        restaurant.location.longitude,
                        ordered[place_idx].location.latitude,
                        ordered[place_idx].location.longitude,
                    ) * TRAVEL_MINUTES_PER_KM
                ))
                current_time += timedelta(minutes=travel)

            continue

        # ── Schedule the next place visit ─────────────────────────────────
        place = ordered[place_idx]
        duration = place.estimated_visit_duration_minutes or 60

        visit_end = current_time + timedelta(minutes=duration)
        if visit_end > day_end:
            break  # no time left for this place

        events.append(VisitEvent(
            start_time=current_time,
            end_time=visit_end,
            place=place,
        ))

        current_time = visit_end

        # Add travel time to next place (if any)
        if place_idx + 1 < len(ordered):
            travel = estimate_travel_minutes(place, ordered[place_idx + 1])
            current_time += timedelta(minutes=travel)

        place_idx += 1

    # ── Insert missed meals after all visits are done ───────────────────────
    remaining_meals = [
        (MealSlot.lunch,  LUNCH_WINDOW),
        (MealSlot.dinner, DINNER_WINDOW),
    ]

    for meal_slot, (window_start, window_end) in remaining_meals:
        if meal_slot.value in meals_inserted:
            continue
        if current_time >= day_end:
            break

        # Schedule the meal at window_start if we're still before it,
        # or at current_time if we've already passed the start
        meal_time = current_time
        if current_time.hour < window_start:
            meal_time = current_time.replace(hour=window_start, minute=0, second=0)
        if meal_time.hour >= window_end or meal_time >= day_end:
            continue

        if events:
            last = events[-1]
            lat = (last.place.location.latitude if isinstance(last, VisitEvent)
                   else last.restaurant.location.latitude)
            lon = (last.place.location.longitude if isinstance(last, VisitEvent)
                   else last.restaurant.location.longitude)
        else:
            lat, lon = _centroid(ordered)

        restaurant = await _find_restaurant(
            meal_slot=meal_slot.value,
            latitude=lat,
            longitude=lon,
            budget_per_person=budget_per_meal,
            preferences=preferences,
        )
        if restaurant:
            events.append(MealEvent(
                time=meal_time,
                meal_slot=meal_slot,
                restaurant=restaurant,
            ))
            current_time = meal_time + timedelta(minutes=MEAL_DURATION_MINUTES)

    return DailySchedule(date=day_date, events=events)


# ── Restaurant lookup via Agent 4 ────────────────────────────────────────────

async def _find_restaurant(
    meal_slot: str,
    latitude: float,
    longitude: float,
    budget_per_person: float | None = None,
    preferences: list[str] | None = None,
) -> RestaurantCandidate | None:
    """
    Call Agent 4 and pick the top-rated candidate.
    Returns None on failure (the scheduler can still continue without a meal).
    """
    try:
        candidates = await call_food_recommender(
            meal_slot=meal_slot,
            latitude=latitude,
            longitude=longitude,
            radius_meters=RESTAURANT_SEARCH_RADIUS,
            budget_per_person=budget_per_person,
            preferences=preferences,
        )
    except Exception as e:
        print(f"⚠️  Agent 4 call failed for {meal_slot}: {e}")
        return None

    if not candidates:
        return None

    # Pick the highest-rated candidate
    best = max(candidates, key=lambda r: r.get("rating") or 0.0)

    return RestaurantCandidate(
        id=best.get("id", "unknown"),
        name=best.get("name", "Unknown"),
        location=best.get("location", {"latitude": latitude, "longitude": longitude}),
        price_level=best.get("price_level"),
        cuisines=best.get("cuisines"),
        rating=best.get("rating"),
        summary=best.get("summary"),
    )
