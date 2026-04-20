from datetime import date, datetime, time
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_TRANSPORT_MODES = {"walking", "driving", "transit", "bicycling"}
DEFAULT_TRANSPORT_MODE = "walking"
EVENT_TYPE_VISIT = "visit"
EVENT_TYPE_TRAVEL = "travel"
EVENT_TYPE_MEAL = "meal"
MEAL_SLOT_LUNCH = "lunch"
MEAL_SLOT_DINNER = "dinner"
WARNING_AGENT4_UNAVAILABLE = "agent4_unavailable_using_synthetic_meal"
WARNING_NO_RESTAURANT_CANDIDATE = "no_restaurant_candidate_found"
WARNING_FALLBACK_TRAVEL = "route_estimation_failed_using_fallback"
WARNING_MEAL_NOT_INSERTED = "meal_not_inserted"
WARNING_PLACE_UNSCHEDULED = "place_unscheduled"


class HealthResponse(BaseModel):
    status: str


class Coordinates(BaseModel):
    """Internal coordinate shape used by existing Agent 3 clients."""

    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    address: str | None = None


class OpeningHoursEntry(BaseModel):
    day_of_week: str = Field(min_length=1)
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
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    location: Location
    estimated_visit_duration_minutes: int = Field(gt=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    category: str | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    summary: str | None = None
    priority_score: float | None = None
    opening_hours: list[OpeningHoursEntry] = Field(default_factory=list)


class DaySchedulingRequest(BaseModel):
    places: list[PlaceCandidate] = Field(min_length=1)
    day_start: datetime
    day_end: datetime
    food_budget_per_day: float | None = Field(default=None, ge=0)
    preferences: list[str] = Field(default_factory=list)
    acceptable_transport_modes: list[str] = Field(default_factory=lambda: [DEFAULT_TRANSPORT_MODE])

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
        unsupported_modes = [
            mode for mode in normalized if mode not in SUPPORTED_TRANSPORT_MODES
        ]
        if unsupported_modes:
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


class RestaurantSummary(BaseModel):
    id: str
    name: str
    location: Location
    price_level: int = Field(ge=1, le=4)
    cuisines: list[str]
    rating: float = Field(ge=0, le=5)
    summary: str


class VisitEvent(BaseModel):
    event_type: Literal["visit"] = EVENT_TYPE_VISIT
    start_time: datetime
    end_time: datetime
    place: PlaceCandidate


class TravelInfo(BaseModel):
    event_type: Literal["travel"] = EVENT_TYPE_TRAVEL
    start_time: datetime
    end_time: datetime
    origin: Location
    destination: Location
    transport_mode: str
    transport_description: str
    estimated_travel_time_minutes: int = Field(ge=0)


class MealEvent(BaseModel):
    event_type: Literal["meal"] = EVENT_TYPE_MEAL
    meal_slot: Literal["lunch", "dinner"]
    start_time: datetime
    end_time: datetime
    restaurant: RestaurantSummary | None = None
    synthetic: bool = False


ScheduleEvent = Annotated[
    VisitEvent | TravelInfo | MealEvent,
    Field(discriminator="event_type"),
]


class DailySchedule(BaseModel):
    date: date
    events: list[ScheduleEvent]


class DaySchedulingResult(BaseModel):
    day_schedule: DailySchedule
    unscheduled_places: list[PlaceCandidate]
    warnings: list[str] = Field(default_factory=list)
