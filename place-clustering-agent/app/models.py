from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None


class OpeningHoursEntry(BaseModel):
    day_of_week: str
    open_time: Optional[str] = None
    close_time: Optional[str] = None


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
    opening_hours: List[OpeningHoursEntry] = Field(default_factory=list)


class ClusteringRequest(BaseModel):
    trip_start: datetime
    trip_end: datetime
    place_candidates: List[PlaceCandidate]


class ClusterDay(BaseModel):
    day_index: int
    total_estimated_visit_minutes: int
    places: List[PlaceCandidate]


class ClusteringResult(BaseModel):
    clustered_place_candidates: List[List[PlaceCandidate]]
    cluster_days: List[ClusterDay]
    warnings: List[str] = Field(default_factory=list)