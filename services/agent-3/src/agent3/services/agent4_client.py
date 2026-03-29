from functools import lru_cache

import httpx

from agent3.core.config import Settings, get_settings
from agent3.core.logging import get_logger
from agent3.models.agent4 import (
    MealRecommendationRequest,
    MealRecommendationResponse,
)


class MealRecommendationError(Exception):
    pass


class Agent4MealClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("agent3.agent4_client", environment=settings.environment)

    def recommend_meal(
        self,
        request: MealRecommendationRequest,
    ) -> MealRecommendationResponse:
        try:
            with httpx.Client(timeout=self._settings.agent4_timeout_seconds) as client:
                response = client.post(
                    f"{self._settings.agent4_base_url.rstrip('/')}/v1/recommend-meal",
                    json=request.model_dump(mode="json"),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            self._logger.warning(
                "meal_recommendation_failed",
                extra={
                    "event": "agent4_meal_recommendation_failed",
                    "agent4_base_url": self._settings.agent4_base_url,
                    "time_of_day": request.time_of_day,
                },
            )
            raise MealRecommendationError("Meal recommendation request failed") from exc

        try:
            return MealRecommendationResponse.model_validate(response.json())
        except ValueError as exc:
            self._logger.warning(
                "meal_recommendation_invalid_response",
                extra={
                    "event": "agent4_meal_recommendation_invalid_response",
                    "agent4_base_url": self._settings.agent4_base_url,
                    "time_of_day": request.time_of_day,
                },
            )
            raise MealRecommendationError("Meal recommendation response was invalid") from exc


@lru_cache(maxsize=1)
def get_agent4_meal_client() -> Agent4MealClient:
    return Agent4MealClient(get_settings())
