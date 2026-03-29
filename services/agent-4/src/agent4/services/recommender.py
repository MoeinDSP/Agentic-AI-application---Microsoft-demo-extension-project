from collections.abc import Iterable
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
    RestaurantCandidate(
        id="osteria-transit-stop",
        name="Osteria Transit Stop",
        location=Coordinates(lat=41.9035, lng=12.4872),
        price_level=3,
        cuisines=["italian", "seafood"],
        rating=4.8,
        summary="Higher-end sit-down lunch with quick seafood specials.",
    ),
    RestaurantCandidate(
        id="curry-corner-roma",
        name="Curry Corner Roma",
        location=Coordinates(lat=41.8965, lng=12.4787),
        price_level=2,
        cuisines=["indian", "vegetarian", "spicy"],
        rating=4.6,
        summary="Warm curries and rice bowls that work well for lunch.",
    ),
]


class RecommenderService:
    def recommend(
        self,
        request: MealRecommendationRequest,
    ) -> MealRecommendationResponse:
        ranked_candidates = sorted(
            self._eligible_candidates(request),
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2].rating,
                item[3],
                item[2].name,
            ),
        )
        return MealRecommendationResponse(
            candidates=[candidate for _, _, candidate, _ in ranked_candidates[:5]]
        )

    def _eligible_candidates(
        self,
        request: MealRecommendationRequest,
    ) -> list[tuple[int, int, RestaurantCandidate, int]]:
        eligible: list[tuple[int, int, RestaurantCandidate, int]] = []
        for candidate in CATALOG:
            distance_meters = self._approximate_distance_meters(
                request.search_center,
                candidate.location,
            )
            if distance_meters > request.search_radius_meters:
                continue

            eligible.append(
                (
                    self._budget_fit_score(
                        candidate.price_level,
                        request.budget_per_meal_per_person,
                    ),
                    self._preference_match_score(candidate, request.preferences),
                    candidate,
                    distance_meters,
                )
            )
        return eligible

    def _budget_fit_score(
        self,
        price_level: int,
        budget_per_meal_per_person: float | None,
    ) -> int:
        if budget_per_meal_per_person is None:
            return 1
        estimated_budget = price_level * 12
        return int(estimated_budget <= budget_per_meal_per_person)

    def _preference_match_score(
        self,
        candidate: RestaurantCandidate,
        preferences: Iterable[str],
    ) -> int:
        searchable_terms = {
            candidate.name.lower(),
            candidate.summary.lower(),
            *[cuisine.lower() for cuisine in candidate.cuisines],
        }
        matches = 0
        for preference in preferences:
            if any(preference in term for term in searchable_terms):
                matches += 1
        return matches

    def _approximate_distance_meters(
        self,
        origin: Coordinates,
        destination: Coordinates,
    ) -> int:
        lat_delta = abs(origin.lat - destination.lat) * 111_000
        lng_delta = abs(origin.lng - destination.lng) * 85_000
        return int(lat_delta + lng_delta)


@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    return RecommenderService()
