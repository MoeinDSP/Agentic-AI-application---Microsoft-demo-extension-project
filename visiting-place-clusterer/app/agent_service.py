from __future__ import annotations

from enum import Enum
from pathlib import Path

from fasta2a import FastA2A
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from starlette.applications import Starlette

from clustering_service import ClusteringService
from environment_service import env
from schemas import Place, PlaceClustererOutput
from utils import FileStore


class AgentEnum(str, Enum):
    PLACE_CLUSTERER = "place_clusterer"


def cluster_places(places: list[Place], num_days: int) -> list[list[Place]]:
    """Group places into geographically coherent day-clusters using K-means.

    Pass the full list of places to visit and the number of trip days. Returns the
    clusters as a list (one per day, capped at the number of places). Every place is
    returned exactly as given and assigned to exactly one cluster — none are added,
    dropped, or modified.
    """
    return ClusteringService.cluster(places, num_days)


class AgentHandle:
    """Runtime wrapper around a built pydantic-ai agent."""

    def __init__(
        self,
        agent_enum: AgentEnum,
        agent: Agent[None, PlaceClustererOutput],
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
            case AgentEnum.PLACE_CLUSTERER:
                return PlaceClustererOutput
            case _:
                raise ValueError(f"Invalid agent: {agent.value}")

    @staticmethod
    def get_tools(agent: AgentEnum) -> list:
        """Return the function tools registered for the given agent."""
        match agent:
            case AgentEnum.PLACE_CLUSTERER:
                return [cluster_places]
            case _:
                raise ValueError(f"Invalid agent: {agent.value}")

    @staticmethod
    def create(
        agent_enum: AgentEnum,
        model: Model,
    ) -> AgentHandle:
        """Build a pydantic-ai agent and return a handle to expose it."""
        agent: Agent[None, PlaceClustererOutput] = Agent(
            model=model,
            system_prompt=AgentService.get_system_prompt(agent_enum),
            tools=AgentService.get_tools(agent_enum),
            output_type=AgentService.get_output_type(agent_enum),
        )
        return AgentHandle(agent_enum=agent_enum, agent=agent)
