from __future__ import annotations

import json

import vertexai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from app.core.config import settings
from app.tools.places_api import search_restaurants

# ── Initialise Vertex AI SDK ──────────────────────────────────────────────────
# Credentials are resolved automatically from GOOGLE_APPLICATION_CREDENTIALS
vertexai.init(
    project=settings.google_cloud_project,
    location=settings.google_cloud_location,
)


# ── ADK Tool ──────────────────────────────────────────────────────────────────
# Must be a plain async function so ADK can auto-generate the JSON schema
# from the type hints and docstring for Gemini's function-calling feature.

async def find_restaurants_tool(
    latitude: float,
    longitude: float,
    radius_meters: int,
    meal_slot: str,
    budget_per_person: float | None = None,
    preferences: list[str] | None = None,
) -> str:
    """
    Search for restaurants near the given coordinates using Google Places API.

    Args:
        latitude:          Latitude of the search centre.
        longitude:         Longitude of the search centre.
        radius_meters:     Search radius in metres (100 – 50 000).
        meal_slot:         One of 'breakfast', 'lunch', 'dinner'.
        budget_per_person: Optional per-person budget in EUR.
        preferences:       Optional list of cuisine or dietary preference strings.

    Returns:
        JSON string containing a list of RestaurantCandidate objects.
    """
    candidates = await search_restaurants(
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        meal_slot=meal_slot,
        budget_per_person=budget_per_person,
        preferences=preferences,
    )
    return json.dumps(candidates, ensure_ascii=False, default=str)


# ── ADK Agent ─────────────────────────────────────────────────────────────────
food_agent = Agent(
    name="food_recommender",
    model=settings.vertex_ai_model,
    description=settings.agent_description,
    instruction=(
        "You are a precision-oriented restaurant recommendation agent "
        "for trip itineraries.\n\n"
        "Rules:\n"
        "- Always call `find_restaurants_tool` with the exact parameters provided.\n"
        "- Return ONLY a JSON object with key `restaurantcandidates` containing "
        "the list returned by the tool — do not add, remove, or modify any field.\n"
        "- Sort candidates: highest rating first; on equal rating prefer lower price_level.\n"
        "- If the tool returns an empty list return: {\"restaurantcandidates\": []}.\n"
        "- Never fabricate restaurant names, ratings, or coordinates."
    ),
    tools=[find_restaurants_tool],
)


# ── Runner & Session service ──────────────────────────────────────────────────
# Both are module-level singletons shared across all A2A worker task executions.

session_service = InMemorySessionService()

runner = Runner(
    agent=food_agent,
    app_name="food_recommender",
    session_service=session_service,
)