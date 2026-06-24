from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from fasta2a import FastA2A
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from starlette.applications import Starlette

from environment_service import env
from places_service import PlacesService
from schemas import FoodRecommenderOutput
from utils import FileStore


class AgentEnum(str, Enum):
    FOOD_RECOMMENDER = "food_recommender"


async def search_restaurants_tool(
    latitude: float,
    longitude: float,
    radius_meters: int,
    meal_slot: str,
    budget_per_person: float | None = None,
    preferences: list[str] | None = None,
) -> str:
    """Search for restaurants near the given coordinates using Google Places API.

    Args:
        latitude: Latitude of the search centre (WGS84 degrees).
        longitude: Longitude of the search centre (WGS84 degrees).
        radius_meters: Search radius in metres (100–50 000).
        meal_slot: One of 'breakfast', 'lunch', 'dinner'.
        budget_per_person: Optional per-person budget in EUR.
        preferences: Optional list of cuisine or dietary preference strings.

    Returns:
        JSON string containing a list of RestaurantCandidate objects.
    """
    candidates = await PlacesService.search_restaurants(
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        meal_slot=meal_slot,
        budget_per_person=budget_per_person,
        preferences=preferences,
    )
    return json.dumps(
        [c.model_dump() for c in candidates], ensure_ascii=False, default=str
    )


class AgentHandle:
    """Runtime wrapper around a built pydantic-ai agent."""

    def __init__(
        self,
        agent_enum: AgentEnum,
        agent: Agent[None, FoodRecommenderOutput],
    ) -> None:
        self.agent_enum = agent_enum
        self.agent = agent

    @property
    def a2a(self) -> FastA2A:
        """Expose the agent as a FastA2A ASGI app for orchestrator integration."""
        return agent_to_a2a(self.agent, **AgentService.get_agent_card(self.agent_enum))

    @property
    def web(self) -> Starlette:
        """Expose the agent as a Starlette ASGI app serving a web chat UI."""
        return self.agent.to_web()


class AgentService:
    """Loads agent configuration and builds pydantic-ai agents."""

    _FILES_DIR = Path(__file__).parent / "files"

    @staticmethod
    def get_system_prompt(agent: AgentEnum) -> str:
        """Return the system prompt for the given agent."""
        data = FileStore.load_yaml(AgentService._FILES_DIR / "system_prompts.yml")
        return data[agent.value]

    @staticmethod
    def get_agent_card(agent: AgentEnum) -> dict:
        """Return the A2A agent card for the given agent."""
        data = FileStore.load_yaml(AgentService._FILES_DIR / "agent_cards.yml")
        card = dict(data[agent.value])
        card["url"] = env.public_url
        return card

    @staticmethod
    def get_output_type(agent: AgentEnum) -> type[BaseModel]:
        """Return the structured output schema for the given agent."""
        match agent:
            case AgentEnum.FOOD_RECOMMENDER:
                return FoodRecommenderOutput
            case _:
                raise ValueError(f"Invalid agent: {agent.value}")

    @staticmethod
    def create(
        agent_enum: AgentEnum,
        model: Model,
    ) -> AgentHandle:
        """Build a pydantic-ai agent and return a handle to expose it."""
        agent: Agent[None, FoodRecommenderOutput] = Agent(
            model=model,
            system_prompt=AgentService.get_system_prompt(agent_enum),
            tools=[search_restaurants_tool],
            output_type=AgentService.get_output_type(agent_enum),
        )
        return AgentHandle(agent_enum=agent_enum, agent=agent)
