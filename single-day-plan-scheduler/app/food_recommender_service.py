from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

import httpx

from environment_service import env


class FoodRecommenderError(Exception):
    """Raised when the food-place-recommender agent cannot be reached or fails."""


class FoodRecommenderClient:
    """A2A client for the food-place-recommender agent.

    Sends a FoodRecommenderInput payload over the A2A JSON-RPC protocol
    (the same protocol the food-place-recommender exposes via fasta2a),
    polls the resulting task to completion, and returns the restaurant
    candidates carried in the task artifacts.
    """

    def __init__(self) -> None:
        self._base_url = env.food_recommender_url.rstrip("/")
        self._timeout = env.food_recommender_timeout_seconds
        self._poll_interval = env.food_recommender_poll_interval_seconds
        self._max_wait = env.food_recommender_max_wait_seconds

    def recommend(
        self,
        *,
        meal_slot: str,
        latitude: float,
        longitude: float,
        search_radius_meters: int,
        budget_per_person: Optional[float],
        preferences: list[str],
    ) -> list[dict[str, Any]]:
        """Return a list of restaurant candidate dicts for the given meal search."""
        payload_text = json.dumps(
            {
                "timeofday": meal_slot,
                "searchcenter": {"latitude": latitude, "longitude": longitude},
                "searchradiusmeters": search_radius_meters,
                "budgetpermealperperson": budget_per_person,
                "preferences": preferences or None,
            }
        )
        message_id = f"plan-scheduler-{uuid.uuid4().hex}"
        send_payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": payload_text}],
                    "kind": "message",
                    "messageId": message_id,
                },
                "configuration": {"acceptedOutputModes": ["application/json"]},
            },
            "id": message_id,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}/", json=send_payload)
                response.raise_for_status()
                task_id = self._extract_task_id(response.json())
                task = self._poll_for_completion(client, task_id)
        except httpx.HTTPError as exc:
            raise FoodRecommenderError(
                f"food-place-recommender request failed: {exc}"
            ) from exc

        return self._extract_restaurants(task)

    def _poll_for_completion(self, client: httpx.Client, task_id: str) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            if time.monotonic() - started > self._max_wait:
                raise FoodRecommenderError("food-place-recommender task timed out")

            response = client.post(
                f"{self._base_url}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {"id": task_id},
                    "id": f"{task_id}-poll",
                },
            )
            response.raise_for_status()
            task = response.json().get("result", {})
            state = task.get("status", {}).get("state")
            if state == "completed":
                return task
            if state in {"failed", "canceled", "rejected"}:
                raise FoodRecommenderError(
                    f"food-place-recommender task ended in state '{state}'"
                )
            time.sleep(self._poll_interval)

    def _extract_task_id(self, payload: dict[str, Any]) -> str:
        task_id = payload.get("result", {}).get("id")
        if not isinstance(task_id, str) or not task_id:
            raise FoodRecommenderError("A2A message/send response did not include a task id")
        return task_id

    def _extract_restaurants(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                data = part.get("data")
                if not isinstance(data, dict):
                    continue
                restaurants = data.get("restaurantcandidates")
                if not isinstance(restaurants, list):
                    restaurants = data.get("restaurants")
                if isinstance(restaurants, list):
                    return [item for item in restaurants if isinstance(item, dict)]
        return []


def recommend_restaurants(
    meal_slot: str,
    latitude: float,
    longitude: float,
    search_radius_meters: int = 1500,
    budget_per_person: Optional[float] = None,
    preferences: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Find restaurants near a location for a meal slot via the food-place-recommender agent.

    Use this to pick a restaurant when inserting a lunch or dinner meal into the day plan.

    Args:
        meal_slot: One of 'breakfast', 'lunch', 'dinner'.
        latitude: Latitude of the search centre (WGS84 degrees).
        longitude: Longitude of the search centre (WGS84 degrees).
        search_radius_meters: Search radius in metres (default 1500).
        budget_per_person: Optional per-person budget in EUR; omit if unknown.
        preferences: Optional list of cuisine or dietary preference strings.

    Returns:
        A dict with key 'restaurants' (a list of restaurant candidate objects, best first,
        each with id, name, location, price_level, cuisines, rating, summary) and key
        'note' describing the outcome. The list is empty when nothing was found.
    """
    client = FoodRecommenderClient()
    try:
        restaurants = client.recommend(
            meal_slot=meal_slot,
            latitude=latitude,
            longitude=longitude,
            search_radius_meters=search_radius_meters,
            budget_per_person=budget_per_person,
            preferences=preferences or [],
        )
    except FoodRecommenderError as exc:
        return {"restaurants": [], "note": f"food_recommender_unavailable: {exc}"}

    if not restaurants:
        return {"restaurants": [], "note": "no_restaurants_found"}
    return {"restaurants": restaurants, "note": f"found {len(restaurants)} restaurants"}
