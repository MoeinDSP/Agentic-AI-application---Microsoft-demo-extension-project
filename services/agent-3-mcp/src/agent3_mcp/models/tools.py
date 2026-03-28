from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str


class Coordinates(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class RouteEstimateRequest(BaseModel):
    origin: Coordinates
    destination: Coordinates
    mode: str = Field(min_length=1)

    @model_validator(mode="after")
    def normalize_mode(self) -> "RouteEstimateRequest":
        self.mode = self.mode.strip().lower()
        return self


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
