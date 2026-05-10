"""
Google ADK agent — Agent 4: Food Recommender.

Uses Google ADK as the agent framework, but routes the underlying LLM
calls through OpenRouter via LiteLLM. This gives access to OpenAI,
Anthropic, Google, Meta, Mistral, and many others behind a single key.
"""
from __future__ import annotations

import json
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from app.core.config import settings
from app.tools.places_api import search_restaurants

# ── OpenRouter credentials for LiteLLM ────────────────────────────────────────
# LiteLLM reads OPENROUTER_API_KEY from the environment automatically.
# The two OR_SITE_URL / OR_APP_NAME variables are optional headers
# OpenRouter uses for usage attribution on its dashboard.
os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
if settings.openrouter_site_url:
    os.environ["OR_SITE_URL"] = settings.openrouter_site_url
if settings.openrouter_app_name:
    os.environ["OR_APP_NAME"] = settings.openrouter_app_name


# ── ADK Tool ──────────────────────────────────────────────────────────────────
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
# `LiteLlm(model=...)` is ADK's adapter for non-Gemini models.
# Whatever string you pass is forwarded to LiteLLM, which routes it
# to the right provider — `openrouter/<provider>/<model>` for OpenRouter.
food_agent = Agent(
    name="food_recommender",
    model=LiteLlm(model=settings.openrouter_model),
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
session_service = InMemorySessionService()
runner = Runner(
    agent=food_agent,
    app_name="food_recommender",
    session_service=session_service,
)
