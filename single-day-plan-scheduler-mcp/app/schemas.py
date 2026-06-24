from __future__ import annotations

from pydantic import BaseModel, Field


class RouteEstimate(BaseModel):
    """Travel estimate between two coordinates for a transport mode."""

    mode: str
    estimated_distance_km: float
    estimated_duration_minutes: int = Field(ge=0)
    notes: list[str] = Field(default_factory=list)


class PlaceDetail(BaseModel):
    """Metadata for a single place id."""

    place_id: str
    display_name: str
    category: str
    summary: str


class PlaceDetails(BaseModel):
    """A batch of place metadata."""

    places: list[PlaceDetail] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
