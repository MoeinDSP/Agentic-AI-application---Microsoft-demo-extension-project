from __future__ import annotations

from fasta2a import FastA2A
from pydantic_ai import Agent
from pydantic_ai.models import Model

from config import AgentEnum, get_agent_card, get_system_prompt


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
        )

    @property
    def a2a(self) -> FastA2A:
        """Returns the Agent converted into a FastA2A ASGI app."""
        return self.agent.to_a2a(**get_agent_card(self.agent_enum))

    @property
    def web(self):
        """Returns the Agent in a web-compatible interface (implementation may vary)."""
        return self.agent.to_web()