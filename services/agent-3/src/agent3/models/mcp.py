from typing import Literal

from pydantic import BaseModel, Field

from agent3.models.plan import Coordinates


class MCPRouteEstimateRequest(BaseModel):
    origin: Coordinates
    destination: Coordinates
    mode: str = Field(min_length=1)


class MCPRouteEstimateResponse(BaseModel):
    mode: str
    estimated_distance_km: float
    estimated_duration_minutes: int
    notes: list[str]


class TravelEstimate(BaseModel):
    source: Literal["mcp", "fallback"]
    mode: str
    estimated_duration_minutes: int = Field(ge=0)
    notes: list[str] = Field(default_factory=list)
