from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MealSlot(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"


class Location(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None


class RestaurantCandidate(BaseModel):
    id: str
    name: str
    location: Location
    price_level: Optional[float] = Field(
        None,
        description="Google Places price scale: 1 (cheap) → 4 (expensive)",
    )
    cuisines: Optional[list[str]] = None
    rating: Optional[float] = Field(None, ge=0, le=5)
    summary: Optional[str] = None


class FoodRecommenderInput(BaseModel):
    timeofday: MealSlot
    searchcenter: Location
    searchradiusmeters: int = Field(1000, ge=100, le=50_000)
    budgetpermealperperson: Optional[float] = None
    preferences: Optional[list[str]] = None


class FoodRecommenderOutput(BaseModel):
    restaurantcandidates: list[RestaurantCandidate]