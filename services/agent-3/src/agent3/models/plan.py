from datetime import time

from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_TRANSPORT_MODES = {"walk", "drive", "transit"}


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


class PlanRequest(BaseModel):
    start_location: StartLocation
    day_start: time
    day_end: time
    transport_preferences: list[str] = Field(default_factory=list)
    places: list[PlaceInput] = Field(min_length=1)

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
        return self


class PlannedStop(BaseModel):
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
