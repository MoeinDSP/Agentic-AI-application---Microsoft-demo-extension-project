from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str


class Coordinates(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class RouteEstimateRequest(BaseModel):
    origin: Coordinates
    destination: Coordinates
    mode: str = Field(min_length=1)
    departure_time: datetime | None = None

    @field_validator("mode")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"walking", "driving", "transit", "bicycling"}:
            raise ValueError("mode must be one of: walking, driving, transit, bicycling")
        return normalized


class RouteEstimateResponse(BaseModel):
    mode: str
    estimated_distance_km: float
    estimated_duration_minutes: int
    notes: list[str]


class PlaceDetailsRequest(BaseModel):
    place_ids: list[str] = Field(min_length=1)


class PlaceDetail(BaseModel):
    place_id: str
    display_name: str
    category: str
    summary: str


class PlaceDetailsResponse(BaseModel):
    places: list[PlaceDetail]
    notes: list[str]
