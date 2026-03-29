from typing import Literal

from pydantic import BaseModel, Field, field_validator

from agent3.models.plan import Coordinates

AGENT4_INVOCATION_MODE_HTTP = "http"
AGENT4_INVOCATION_MODE_A2A = "a2a"
SUPPORTED_AGENT4_INVOCATION_MODES = {
    AGENT4_INVOCATION_MODE_HTTP,
    AGENT4_INVOCATION_MODE_A2A,
}


class MealRecommendationRequest(BaseModel):
    time_of_day: Literal["breakfast", "lunch", "dinner"] = "lunch"
    search_center: Coordinates
    search_radius_meters: int = Field(default=1000, gt=0)
    budget_per_meal_per_person: float | None = Field(default=None, ge=0)
    preferences: list[str] = Field(default_factory=list)

    @field_validator("preferences")
    @classmethod
    def normalize_preferences(cls, value: list[str]) -> list[str]:
        return [item.strip().lower() for item in value if item.strip()]


class RestaurantCandidate(BaseModel):
    id: str
    name: str
    location: Coordinates
    price_level: int = Field(ge=1, le=4)
    cuisines: list[str]
    rating: float = Field(ge=0, le=5)
    summary: str


class MealRecommendationResponse(BaseModel):
    candidates: list[RestaurantCandidate]


class Agent4A2ARequest(BaseModel):
    request_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    input: MealRecommendationRequest
    accepted_content_types: list[str] = Field(default_factory=lambda: ["application/json"])


class Agent4A2AResponse(BaseModel):
    request_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    result_type: str = Field(min_length=1)
    output: MealRecommendationResponse
    notes: list[str] = Field(default_factory=list)
