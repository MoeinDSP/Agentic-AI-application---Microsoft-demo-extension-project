from __future__ import annotations

from pydantic import BaseModel, Field


class Place(BaseModel):
    """A place to visit, as produced by the visiting place recommender."""

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
        description="A short rationale explaining why this place was recommended."
    )
    rank: int = Field(
        description=(
            "The position of the place in the recommendations list. Starts at 0 for "
            "the highest-priority recommendation and increases by 1 with no gaps."
        ),
        ge=0,
    )


class PlaceCluster(BaseModel):
    """A geographically coherent group of places intended for a single trip day."""

    day: int = Field(
        description=(
            "The 1-based day index this cluster is planned for. Starts at 1 and "
            "increases by 1 with no gaps."
        ),
        ge=1,
    )
    places: list[Place] = Field(
        description=(
            "The places grouped together for this day, kept exactly as received "
            "from the recommender. Order within the day is not significant."
        )
    )


class PlaceClustererOutput(BaseModel):
    """Structured output for the visiting place clusterer agent."""

    description: str = Field(
        description=(
            "A natural-language summary of the clustering outcome. When `clusters` "
            "is non-empty, briefly describe how the places were grouped across the "
            "trip days. When `clusters` is empty, clearly state which required "
            "information is missing (e.g. the list of places or the trip dates)."
        )
    )
    clusters: list[PlaceCluster] = Field(
        default_factory=list,
        description=(
            "One entry per trip day, each holding the places assigned to that day. "
            "Every input place appears in exactly one cluster, unchanged. Empty when "
            "required information is missing."
        ),
    )
