from __future__ import annotations

from fasta2a import FastA2A
from pydantic_ai import Agent
from pydantic_ai.models import Model

from config import AgentEnum, get_agent_card, get_system_prompt, get_output_type


class AgentFactory:
    def __init__(
        self,
        agent_enum: AgentEnum,
        model: Model,
        toolsets: list | None = None,
    ):
        self.agent_enum = agent_enum
        self.agent = Agent(
            model=model,
            system_prompt=get_system_prompt(self.agent_enum),
            toolsets=toolsets or [],
            output_type=get_output_type(self.agent_enum),
        )

    @property
    def a2a(self) -> FastA2A:
        """Expose the agent as a FastA2A ASGI app for orchestrator integration."""
        return self.agent.to_a2a(**get_agent_card(self.agent_enum))

    @property
    def web(self):
        """Expose the agent as a Starlette ASGI app serving a web chat UI."""
        return self.agent.to_web()
