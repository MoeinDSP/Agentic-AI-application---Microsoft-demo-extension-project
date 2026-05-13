from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from math import ceil
from typing import Optional, Union

from pydantic import BaseModel, Field


# ── Shared data models ────────────────────────────────────────────────────────

class MealSlot(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"


class Location(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None


class BudgetConstraint(BaseModel):
    total_budget: float
    currency: str = "EUR"


class PlaceCandidate(BaseModel):
    """Compatible with Agent 2's PlaceCandidate (includes opening_hours)."""
    id: str
    name: str
    location: Location
    estimated_visit_duration_minutes: int = Field(default=60, ge=0)
    estimated_cost: Optional[float] = None
    category: Optional[str] = None
    rating: Optional[float] = None
    summary: Optional[str] = None
    priority_score: Optional[float] = Field(default=0.0)
    opening_hours: Optional[list[dict]] = Field(default_factory=list)


class RestaurantCandidate(BaseModel):
    id: str
    name: str
    location: Location
    price_level: Optional[float] = None
    cuisines: Optional[list[str]] = None
    rating: Optional[float] = None
    summary: Optional[str] = None


# ── Event models ──────────────────────────────────────────────────────────────

class VisitEvent(BaseModel):
    start_time: datetime
    end_time: datetime
    place: PlaceCandidate


class MealEvent(BaseModel):
    time: datetime
    meal_slot: MealSlot
    restaurant: RestaurantCandidate


class DailySchedule(BaseModel):
    date: date
    events: list[Union[VisitEvent, MealEvent]]


# ── Orchestrator I/O ──────────────────────────────────────────────────────────

class TripRequest(BaseModel):
    city: str
    trip_start: datetime
    trip_end: datetime
    location: Optional[Location] = None
    budget: Optional[BudgetConstraint] = None
    trip_reason: Optional[str] = None
    preferences: Optional[list[str]] = None


class DerivedParams(BaseModel):
    """Computed by the orchestrator from the user request."""
    num_days: int
    activity_budget: Optional[float] = None
    food_budget_total: Optional[float] = None
    food_budget_per_day: Optional[float] = None

    @classmethod
    def from_request(
        cls,
        req: TripRequest,
        activity_ratio: float = 0.70,
        food_ratio: float = 0.30,
    ) -> DerivedParams:
        delta = req.trip_end - req.trip_start
        num_days = max(1, ceil(delta.total_seconds() / 86400))

        if req.budget:
            activity_budget = activity_ratio * req.budget.total_budget
            food_total = food_ratio * req.budget.total_budget
            food_per_day = food_total / num_days
        else:
            activity_budget = None
            food_total = None
            food_per_day = None

        return cls(
            num_days=num_days,
            activity_budget=activity_budget,
            food_budget_total=food_total,
            food_budget_per_day=food_per_day,
        )


class TripItinerary(BaseModel):
    """Final output of the orchestrator."""
    request: TripRequest
    derived: DerivedParams
    daily_schedules: list[DailySchedule]
    warnings: list[str] = Field(default_factory=list)
