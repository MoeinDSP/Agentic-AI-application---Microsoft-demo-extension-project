from typing import List, Optional
from pydantic import BaseModel, Field

class Location(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    address: Optional[str] = Field(None, description="Formatted street address")

class FoodRecommendationRequest(BaseModel):
    time_of_day: str = Field(..., description="Meal slot (breakfast, lunch, dinner) or ISO datetime")
    search_center: Location = Field(..., description="Geographic center for the search")
    search_radius_meters: int = Field(..., description="Search radius in meters")
    budget_per_meal_per_person: Optional[float] = None
    preferences: Optional[List[str]] = None

class RestaurantCandidate(BaseModel):
    id: str = Field(..., description="Unique identifier for the restaurant")
    name: str = Field(..., description="Name of the restaurant")
    location: Location = Field(..., description="Geographic location and address")
    price_level: Optional[float] = Field(None, description="Price level indicator")
    cuisines: Optional[List[str]] = Field(None, description="List of cuisine types offered")
    rating: Optional[float] = Field(None, description="Aggregate user rating")
    summary: Optional[str] = Field(None, description="Short description of why this is a good fit")