import time
import uuid
from functools import lru_cache
from typing import Any

import httpx

from agent3.core.config import Settings, get_settings
from agent3.core.logging import get_logger
from agent3.models.agent4 import (
    AGENT4_INVOCATION_MODE_A2A,
    MealRecommendationRequest,
    MealRecommendationResponse,
    RestaurantCandidate,
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
            self._log_failure("agent4_http_request_failed", request)
            raise MealRecommendationError("Meal recommendation request failed") from exc

        try:
            return MealRecommendationResponse.model_validate(response.json())
        except ValueError as exc:
            self._log_failure("agent4_http_response_invalid", request)
            raise MealRecommendationError("Meal recommendation response was invalid") from exc

    def _recommend_meal_via_a2a(
        self,
        request: MealRecommendationRequest,
    ) -> MealRecommendationResponse:
        message_id = f"agent3-{uuid.uuid4().hex}"
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": self._build_prompt(request),
                        }
                    ],
                    "kind": "message",
                    "messageId": message_id,
                },
                "configuration": {
                    "acceptedOutputModes": ["application/json"],
                },
            },
            "id": message_id,
        }

        try:
            with httpx.Client(timeout=self._settings.agent4_timeout_seconds) as client:
                response = client.post(
                    f"{self._settings.agent4_base_url.rstrip('/')}/",
                    json=payload,
                )
                response.raise_for_status()
                task_id = self._extract_task_id(response.json())
                task = self._poll_for_task_completion(client, task_id, request)
        except httpx.HTTPError as exc:
            self._log_failure("agent4_a2a_request_failed", request)
            raise MealRecommendationError("Meal recommendation request failed") from exc

        try:
            return self._parse_a2a_task_result(task)
        except ValueError as exc:
            self._log_failure("agent4_a2a_response_invalid", request)
            raise MealRecommendationError("Meal recommendation response was invalid") from exc

    def _poll_for_task_completion(
        self,
        client: httpx.Client,
        task_id: str,
        request: MealRecommendationRequest,
    ) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            if time.monotonic() - started > self._settings.agent4_max_wait_seconds:
                self._log_failure("agent4_a2a_timeout", request)
                raise MealRecommendationError("Meal recommendation request timed out")

            response = client.post(
                f"{self._settings.agent4_base_url.rstrip('/')}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {"id": task_id},
                    "id": f"{task_id}-poll",
                },
            )
            response.raise_for_status()
            payload = response.json()
            task = payload.get("result", {})
            state = task.get("status", {}).get("state")
            if state == "completed":
                return task
            if state in {"failed", "canceled"}:
                self._log_failure(f"agent4_a2a_{state}", request)
                raise MealRecommendationError("Meal recommendation task failed")

            time.sleep(self._settings.agent4_poll_interval_seconds)

    def _extract_task_id(self, payload: dict[str, Any]) -> str:
        task_id = payload.get("result", {}).get("id")
        if not task_id or not isinstance(task_id, str):
            raise ValueError("A2A message/send response did not include a task id")
        return task_id

    def _parse_a2a_task_result(self, task: dict[str, Any]) -> MealRecommendationResponse:
        restaurants = self._extract_restaurants(task)
        return MealRecommendationResponse(
            candidates=[
                RestaurantCandidate.model_validate(
                    {
                        "id": restaurant.get("id"),
                        "name": restaurant.get("name"),
                        "location": {
                            "lat": restaurant.get("location", {}).get("latitude"),
                            "lng": restaurant.get("location", {}).get("longitude"),
                        },
                        "price_level": self._normalize_price_level(
                            restaurant.get("price_level")
                        ),
                        "cuisines": restaurant.get("cuisines") or ["unknown"],
                        "rating": restaurant.get("rating") or 0,
                        "summary": restaurant.get("summary")
                        or restaurant.get("location", {}).get("address")
                        or "No summary provided.",
                    }
                )
                for restaurant in restaurants
            ]
        )

    def _extract_restaurants(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                data = part.get("data")
                if not isinstance(data, dict):
                    continue
                restaurants = data.get("restaurants")
                if isinstance(restaurants, list):
                    return [item for item in restaurants if isinstance(item, dict)]
        return []

    def _normalize_price_level(self, value: object) -> int:
        if value is None:
            return 1
        if isinstance(value, (int, float)):
            normalized = int(round(float(value)))
            return min(max(normalized, 1), 4)
        return 1

    def _build_prompt(self, request: MealRecommendationRequest) -> str:
        preferences = ", ".join(request.preferences) if request.preferences else "none"
        budget = (
            f"{request.budget_per_meal_per_person:.2f} EUR per person"
            if request.budget_per_meal_per_person is not None
            else "no explicit budget"
        )
        return (
            "Recommend nearby restaurants for a meal search.\n"
            f"Meal slot: {request.time_of_day}.\n"
            f"Search center latitude: {request.search_center.lat}.\n"
            f"Search center longitude: {request.search_center.lng}.\n"
            f"Search radius meters: {request.search_radius_meters}.\n"
            f"Budget per person: {budget}.\n"
            f"Preferences: {preferences}.\n"
            "Return the best restaurant matches."
        )

    def _log_failure(self, event: str, request: MealRecommendationRequest) -> None:
        self._logger.warning(
            "meal_recommendation_failed",
            extra={
                "event": event,
                "agent4_base_url": self._settings.agent4_base_url,
                "time_of_day": request.time_of_day,
                "invocation_mode": self._settings.agent4_invocation_mode,
            },
        )


@lru_cache(maxsize=1)
def get_agent4_meal_client() -> Agent4MealClient:
    return Agent4MealClient(get_settings())
