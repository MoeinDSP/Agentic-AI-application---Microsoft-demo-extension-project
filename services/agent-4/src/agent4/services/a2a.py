from functools import lru_cache

from agent4.core.config import Settings, get_settings
from agent4.models.a2a import (
    AgentAuth,
    AgentCard,
    AgentEndpoint,
    AgentSkill,
)


class AgentCardService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_agent_card(self) -> AgentCard:
        return AgentCard(
            name="agent-4",
            description=(
                "Deterministic mock meal recommendation agent for lunch-oriented "
                "restaurant discovery."
            ),
            version=self._settings.service_version,
            base_url=self._settings.public_base_url.rstrip("/"),
            interaction_mode="a2a-ready-http",
            endpoints=[
                AgentEndpoint(
                    path=self._settings.agent_card_path,
                    method="GET",
                    description="Agent discovery metadata for orchestrators.",
                ),
                AgentEndpoint(
                    path="/v1/recommend-meal",
                    method="POST",
                    description="Direct typed meal recommendation API.",
                ),
            ],
            skills=[
                AgentSkill(
                    id="meal-recommendation",
                    name="Meal Recommendation",
                    description=(
                        "Returns deterministic ranked restaurant candidates for "
                        "mock lunch recommendation requests."
                    ),
                    tags=["food", "lunch", "mock", "local-first"],
                )
            ],
            content_types=["application/json"],
            auth=AgentAuth(
                mode="none",
                notes="Local-first MVP. No authentication is required yet.",
            ),
            notes=[
                (
                    "This agent card describes an A2A-ready HTTP boundary, "
                    "not a full A2A spec implementation."
                ),
                "Agent 4 remains a deterministic mock recommendation service.",
            ],
        )


@lru_cache(maxsize=1)
def get_agent_card_service() -> AgentCardService:
    return AgentCardService(get_settings())
