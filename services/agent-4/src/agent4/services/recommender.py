from functools import lru_cache

from agent4.models.meal import (
    Coordinates,
    MealRecommendationRequest,
    MealRecommendationResponse,
    RestaurantCandidate,
)

CATALOG = [
    RestaurantCandidate(
        id="trattoria-della-luce",
        name="Trattoria della Luce",
        location=Coordinates(lat=41.8991, lng=12.4828),
        price_level=2,
        cuisines=["italian", "pasta", "roman"],
        rating=4.7,
        summary="Classic Roman lunch menu with quick pasta dishes.",
    ),
    RestaurantCandidate(
        id="verdura-pranzo-bar",
        name="Verdura Pranzo Bar",
        location=Coordinates(lat=41.9012, lng=12.4794),
        price_level=2,
        cuisines=["vegetarian", "salads", "mediterranean"],
        rating=4.5,
        summary="Vegetable-forward lunch plates with a fast casual format.",
    ),
    RestaurantCandidate(
        id="mercato-panini",
        name="Mercato Panini",
        location=Coordinates(lat=41.8978, lng=12.4851),
        price_level=1,
        cuisines=["sandwiches", "italian", "street-food"],
        rating=4.3,
        summary="Budget-friendly panini and takeaway lunch options.",
    ),
]


class RecommenderService:
    def recommend(
        self,
        request: MealRecommendationRequest,
    ) -> MealRecommendationResponse:
        # Placeholder scaffold response. Ranking logic is added separately.
        return MealRecommendationResponse(candidates=CATALOG[:3])


@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    return RecommenderService()
