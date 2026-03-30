from typing import List, Optional
from pydantic import BaseModel, Field

class Location(BaseModel):
    """
    Represents geographic coordinates and an optional address.
    Used for both the search center input and the restaurant output.
    """
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    address: Optional[str] = Field(None, description="Formatted street address")

class RestaurantCandidate(BaseModel):
    """
    The standardized output model for a recommended restaurant.
    This exact structure is expected by Agent 3 (Daily Scheduler).
    """
    id: str = Field(
        ..., 
        description="Unique identifier for the restaurant (e.g., Google Places ID)"
    )
    name: str = Field(
        ..., 
        description="Name of the restaurant"
    )
    location: Location = Field(
        ..., 
        description="Geographic location and address of the restaurant"
    )
    price_level: Optional[float] = Field(
        None, 
        description="Price level indicator (e.g., 1.0 for cheap, 4.0 for expensive)"
    )
    cuisines: Optional[List[str]] = Field(
        None, 
        description="List of cuisine types offered (e.g., ['Italian', 'Seafood'])"
    )
    rating: Optional[float] = Field(
        None, 
        description="Aggregate user rating (e.g., 1.0 to 5.0)"
    )
    summary: Optional[str] = Field(
        None, 
        description="Short description or AI-generated summary of why this is a good fit"
    )