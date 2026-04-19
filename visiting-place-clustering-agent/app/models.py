from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None


class PlaceCandidate(BaseModel):
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


class ClusteringRequest(BaseModel):
    trip_start: datetime
    trip_end: datetime
    place_candidates: list[PlaceCandidate]


class ClusteringResponse(BaseModel):
    clustered_place_candidates: list[list[PlaceCandidate]]
