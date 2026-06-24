from __future__ import annotations

from enum import Enum
from pathlib import Path

from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel
from starlette.applications import Starlette

from environment_service import env
from schemas import DaySchedulingResult
from utils import FileStore


class AgentEnum(str, Enum):
    PLAN_SCHEDULER = "plan_scheduler"


class AgentHandle:
    """Runtime wrapper around a built Google ADK agent."""

    def __init__(self, agent_enum: AgentEnum, agent: LlmAgent) -> None:
        self.agent_enum = agent_enum
        self.agent = agent

    @property
    def a2a(self) -> Starlette:
        """Expose the agent as an A2A ASGI app for orchestrator integration."""
        return to_a2a(
            self.agent,
            agent_card=AgentService.get_agent_card(self.agent_enum),
        )

    @property
    def web(self) -> Starlette:
        """Web chat UI is not provided by ADK at the per-agent ASGI level."""
        raise NotImplementedError(
            "single-day-plan-scheduler only supports A2A mode; "
            "use the Google ADK `adk web` CLI for an interactive UI."
        )


class AgentService:
    """Loads agent configuration and builds Google ADK agents."""

    _FILES_DIR = Path(__file__).parent / "files"

    @staticmethod
    def get_system_prompt(agent: AgentEnum) -> str:
        """Return the system prompt (instruction) for the given agent."""
        data = FileStore.load_yaml(AgentService._FILES_DIR / "system_prompts.yml")
        return data[agent.value]

    @staticmethod
    def get_agent_card(agent: AgentEnum) -> AgentCard:
        """Return the A2A agent card for the given agent."""
        data = FileStore.load_yaml(AgentService._FILES_DIR / "agent_cards.yml")
        card = dict(data[agent.value])
        skills = [AgentSkill(**skill) for skill in card.get("skills", [])]
        return AgentCard(
            name=card["name"],
            description=card["description"],
            version=card["version"],
            url=env.public_url,
            provider=card.get("provider"),
            capabilities=AgentCapabilities(),
            default_input_modes=card.get("input_modes", ["application/json", "text/plain"]),
            default_output_modes=card.get("output_modes", ["application/json"]),
            skills=skills,
        )

    @staticmethod
    def get_output_type(agent: AgentEnum) -> type[BaseModel]:
        """Return the structured output schema for the given agent."""
        match agent:
            case AgentEnum.PLAN_SCHEDULER:
                return DaySchedulingResult
            case _:
                raise ValueError(f"Invalid agent: {agent.value}")

    @staticmethod
    def create(
        agent_enum: AgentEnum,
        model: LiteLlm | str,
        toolsets: list | None = None,
        tools: list | None = None,
    ) -> AgentHandle:
        """Build a Google ADK agent and return a handle to expose it."""
        agent = LlmAgent(
            model=model,
            name=agent_enum.value,
            description=AgentService.get_agent_card(agent_enum).description,
            instruction=AgentService.get_system_prompt(agent_enum),
            tools=[*(tools or []), *(toolsets or [])],
            output_schema=AgentService.get_output_type(agent_enum),
            output_key="day_plan",
        )
        return AgentHandle(agent_enum=agent_enum, agent=agent)
