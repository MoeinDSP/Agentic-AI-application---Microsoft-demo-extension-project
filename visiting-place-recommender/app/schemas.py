from __future__ import annotations

from pydantic import BaseModel, Field


class Place(BaseModel):
    """A recommended place to visit."""

    name: str = Field(
        description="The display name of the place as it appears on Google Maps."
    )
    place_url: str = Field(
        description="A Google Maps link to the place's main listing."
    )
    photos_url: str = Field(description="A Google Maps link to photos of the place.")
    reviews_url: str = Field(description="A Google Maps link to reviews of the place.")
    lat: float = Field(
        description="The latitude in degrees (WGS84), in the range [-90.0, +90.0].",
        ge=-90.0,
        le=90.0,
    )
    lng: float = Field(
        description="The longitude in degrees (WGS84), in the range [-180.0, +180.0].",
        ge=-180.0,
        le=180.0,
    )
    description: str = Field(
        description=(
            "A short rationale explaining why this place was selected and placed "
            "at this rank, grounded in the user's stated preferences, budget, and "
            "constraints."
        )
    )
    rank: int = Field(
        description=(
            "The position of the place in the recommendations list. Starts at 0 for "
            "the highest-priority recommendation and increases by 1 with no gaps."
        ),
        ge=0,
    )


class PlaceRecommenderOutput(BaseModel):
    """Structured output for the place recommender agent."""

    description: str = Field(
        description=(
            "A natural-language summary of the recommendation outcome. When "
            "`places` is non-empty, briefly describe the overall selection rationale. "
            "When `places` is empty, clearly state which required information is "
            "missing (e.g. destination city, type of experience) or why no suitable "
            "places were found."
        )
    )
    places: list[Place] = Field(
        default_factory=list,
        min_length=0,
        max_length=8,
        description=(
            "The ranked list of recommended places to visit in the destination "
            "city. Empty when required information is missing or no suitable "
            "matches were found."
        ),
    )
