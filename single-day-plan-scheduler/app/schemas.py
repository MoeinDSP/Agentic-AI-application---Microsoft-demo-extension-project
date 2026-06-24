from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_TRANSPORT_MODES = {"walking", "driving", "transit", "bicycling"}
DEFAULT_TRANSPORT_MODE = "walking"


class TransportMode(str, Enum):
    walking = "walking"
    driving = "driving"
    transit = "transit"
    bicycling = "bicycling"


class MealSlot(str, Enum):
    lunch = "lunch"
    dinner = "dinner"


class EventType(str, Enum):
    visit = "visit"
    travel = "travel"
    meal = "meal"


class Location(BaseModel):
    """Geographic coordinates with an optional human-readable address."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    address: Optional[str] = None


class OpeningHoursEntry(BaseModel):
    """Opening window for a place on a given weekday."""

    day_of_week: str = Field(min_length=1, description="Lowercase weekday name, e.g. 'monday'.")
    open_time: time
    close_time: time

    @field_validator("day_of_week")
    @classmethod
    def normalize_day_of_week(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def validate_opening_window(self) -> "OpeningHoursEntry":
        if self.close_time <= self.open_time:
            raise ValueError("close_time must be later than open_time")
        return self


class PlaceCandidate(BaseModel):
    """A candidate place to visit, as produced by the upstream recommender/clusterer."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    location: Location
    estimated_visit_duration_minutes: int = Field(gt=0)
    estimated_cost: Optional[float] = Field(default=None, ge=0)
    category: Optional[str] = None
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    summary: Optional[str] = None
    priority_score: Optional[float] = Field(
        default=None,
        description="Higher means the place should be scheduled earlier / preferred.",
    )
    opening_hours: list[OpeningHoursEntry] = Field(default_factory=list)


class RestaurantSummary(BaseModel):
    """A restaurant chosen for a meal slot (sourced from the food recommender)."""

    id: str
    name: str
    location: Location
    price_level: Optional[int] = Field(default=None, ge=1, le=4)
    cuisines: list[str] = Field(default_factory=list)
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    summary: Optional[str] = None


class ScheduleEvent(BaseModel):
    """A single entry in the day's timeline.

    `event_type` selects which of the optional fields are populated:
      - "visit":  set `place`.
      - "travel": set `origin`, `destination`, `transport_mode`,
                  `transport_description`, `estimated_travel_time_minutes`.
      - "meal":   set `meal_slot`, `restaurant` (or leave null and set `synthetic`).
    Fields that do not apply to the event_type are left null.
    """

    event_type: Literal["visit", "travel", "meal"]
    start_time: datetime
    end_time: datetime

    # visit
    place: Optional[PlaceCandidate] = None

    # travel
    origin: Optional[Location] = None
    destination: Optional[Location] = None
    transport_mode: Optional[str] = None
    transport_description: Optional[str] = None
    estimated_travel_time_minutes: Optional[int] = Field(default=None, ge=0)

    # meal
    meal_slot: Optional[Literal["lunch", "dinner"]] = None
    restaurant: Optional[RestaurantSummary] = None
    synthetic: Optional[bool] = Field(
        default=None,
        description="True when no real restaurant could be found and a placeholder meal was inserted.",
    )


class DailySchedule(BaseModel):
    date: date
    events: list[ScheduleEvent] = Field(default_factory=list)


class DaySchedulingRequest(BaseModel):
    """Input schema expected by the single-day plan scheduler agent."""

    places: list[PlaceCandidate] = Field(min_length=1)
    day_start: datetime
    day_end: datetime
    food_budget_per_day: Optional[float] = Field(default=None, ge=0)
    preferences: list[str] = Field(default_factory=list)
    acceptable_transport_modes: list[str] = Field(
        default_factory=lambda: [DEFAULT_TRANSPORT_MODE]
    )

    @field_validator("preferences", mode="before")
    @classmethod
    def normalize_optional_preferences(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("preferences must be a list")
        return list(dict.fromkeys(item.strip().lower() for item in value if item.strip()))

    @field_validator("acceptable_transport_modes")
    @classmethod
    def normalize_transport_modes(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value if item.strip()]
        unsupported = [m for m in normalized if m not in SUPPORTED_TRANSPORT_MODES]
        if unsupported:
            raise ValueError(
                "acceptable_transport_modes must contain only: "
                "walking, driving, transit, bicycling"
            )
        return list(dict.fromkeys(normalized)) or [DEFAULT_TRANSPORT_MODE]

    @model_validator(mode="after")
    def validate_day_window(self) -> "DaySchedulingRequest":
        if self.day_end <= self.day_start:
            raise ValueError("day_end must be later than day_start")
        return self


class DaySchedulingResult(BaseModel):
    """Structured output for the single-day plan scheduler agent."""

    description: str = Field(
        description=(
            "A natural-language summary of the produced day plan: how many places were "
            "scheduled, which meals were inserted, and any notable trade-offs. When no "
            "viable plan could be built, clearly state why."
        )
    )
    day_schedule: DailySchedule = Field(
        description="The ordered list of visit, travel, and meal events for the day."
    )
    unscheduled_places: list[PlaceCandidate] = Field(
        default_factory=list,
        description="Places that could not be fitted into the day window or were closed.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Machine-readable notes about fallbacks, dropped places, or missing data.",
    )
