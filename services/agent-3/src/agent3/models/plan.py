from datetime import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_TRANSPORT_MODES = {"walk", "drive", "transit"}
STOP_TYPE_PLACE = "place"
STOP_TYPE_LUNCH = "lunch"
DROP_REASON_INSUFFICIENT_TIME = "insufficient_time"
DROP_REASON_CLOSED_AT_ARRIVAL = "closed_at_arrival"
DROP_REASON_CLOSES_BEFORE_VISIT_ENDS = "closes_before_visit_ends"
LUNCH_INSERTED_NOTE = "lunch_inserted"
LUNCH_NOT_INSERTED_NOTE = "lunch_not_inserted"


class HealthResponse(BaseModel):
    status: str


class Coordinates(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class StartLocation(Coordinates):
    name: str | None = None


class PlaceInput(Coordinates):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    estimated_duration_minutes: int = Field(gt=0)
    priority: int = Field(ge=1, le=5)
    opens_at: time | None = None
    closes_at: time | None = None

    @model_validator(mode="after")
    def validate_opening_window(self) -> "PlaceInput":
        if (
            self.opens_at is not None
            and self.closes_at is not None
            and self.closes_at <= self.opens_at
        ):
            raise ValueError("closes_at must be later than opens_at")
        return self


class PlanRequest(BaseModel):
    start_location: StartLocation
    day_start: time
    day_end: time
    transport_preferences: list[str] = Field(default_factory=list)
    places: list[PlaceInput] = Field(min_length=1)
    lunch_required: bool = False
    lunch_time_window_start: time = time(hour=12, minute=0)
    lunch_time_window_end: time = time(hour=14, minute=0)
    lunch_duration_minutes: int = Field(default=60, gt=0)

    @field_validator("transport_preferences")
    @classmethod
    def normalize_transport_preferences(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value if item.strip()]
        unsupported_modes = [
            mode for mode in normalized if mode not in SUPPORTED_TRANSPORT_MODES
        ]
        if unsupported_modes:
            raise ValueError(
                "transport_preferences must contain only supported modes: "
                "walk, drive, transit"
            )
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_day_window(self) -> "PlanRequest":
        if self.day_end <= self.day_start:
            raise ValueError("day_end must be later than day_start")
        if self.lunch_time_window_end <= self.lunch_time_window_start:
            raise ValueError(
                "lunch_time_window_end must be later than lunch_time_window_start"
            )
        return self


class PlannedStop(BaseModel):
    stop_type: Literal["place", "lunch"] = STOP_TYPE_PLACE
    place_id: str
    place_name: str
    sequence: int
    arrival_time: time
    start_time: time
    end_time: time
    travel_minutes_from_previous: int = Field(ge=0)
    estimated_duration_minutes: int


class DroppedPlace(BaseModel):
    place_id: str
    reason: str


class PlanResponse(BaseModel):
    ordered_stops: list[PlannedStop]
    dropped_places: list[DroppedPlace]
    notes: list[str]
    feasibility: bool
    selected_transport_mode: str
    total_travel_minutes: int = Field(ge=0)
    total_visit_minutes: int = Field(ge=0)
