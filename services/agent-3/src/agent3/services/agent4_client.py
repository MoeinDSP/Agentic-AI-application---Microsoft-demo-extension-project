import uuid
from functools import lru_cache

import httpx

from agent3.core.config import Settings, get_settings
from agent3.core.logging import get_logger
from agent3.models.agent4 import (
    AGENT4_INVOCATION_MODE_A2A,
    Agent4A2ARequest,
    Agent4A2AResponse,
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
        if self._settings.agent4_invocation_mode == AGENT4_INVOCATION_MODE_A2A:
            return self._recommend_meal_via_a2a(request)
        return self._recommend_meal_via_http(request)

    def _recommend_meal_via_http(
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
                    "invocation_mode": self._settings.agent4_invocation_mode,
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
                    "invocation_mode": self._settings.agent4_invocation_mode,
                },
            )
            raise MealRecommendationError("Meal recommendation response was invalid") from exc

    def _recommend_meal_via_a2a(
        self,
        request: MealRecommendationRequest,
    ) -> MealRecommendationResponse:
        a2a_request = Agent4A2ARequest(
            request_id=f"agent3-{uuid.uuid4().hex}",
            action="recommend_meal",
            input=request,
        )
        try:
            with httpx.Client(timeout=self._settings.agent4_timeout_seconds) as client:
                response = client.post(
                    f"{self._settings.agent4_base_url.rstrip('/')}/a2a",
                    json=a2a_request.model_dump(mode="json"),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            self._logger.warning(
                "meal_recommendation_failed",
                extra={
                    "event": "agent4_meal_recommendation_failed",
                    "agent4_base_url": self._settings.agent4_base_url,
                    "time_of_day": request.time_of_day,
                    "invocation_mode": self._settings.agent4_invocation_mode,
                },
            )
            raise MealRecommendationError("Meal recommendation request failed") from exc

        try:
            payload = Agent4A2AResponse.model_validate(response.json())
        except ValueError as exc:
            self._logger.warning(
                "meal_recommendation_invalid_response",
                extra={
                    "event": "agent4_meal_recommendation_invalid_response",
                    "agent4_base_url": self._settings.agent4_base_url,
                    "time_of_day": request.time_of_day,
                    "invocation_mode": self._settings.agent4_invocation_mode,
                },
            )
            raise MealRecommendationError("Meal recommendation response was invalid") from exc

        return payload.output


@lru_cache(maxsize=1)
def get_agent4_meal_client() -> Agent4MealClient:
    return Agent4MealClient(get_settings())
