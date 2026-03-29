from functools import lru_cache

from agent4.core.config import Settings, get_settings
from agent4.models.a2a import (
    A2ARequest,
    A2AResponse,
    AgentAuth,
    AgentCard,
    AgentEndpoint,
    AgentSkill,
)
from agent4.services.recommender import RecommenderService, get_recommender_service


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
                AgentEndpoint(
                    path=self._settings.a2a_path,
                    method="POST",
                    description="Minimal A2A-shaped meal recommendation endpoint.",
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


class A2AService:
    def __init__(self, recommender: RecommenderService) -> None:
        self._recommender = recommender

    def handle_request(self, request: A2ARequest) -> A2AResponse:
        if request.action != "recommend_meal":
            raise ValueError("action must be recommend_meal")

        output = self._recommender.recommend(request.input)
        return A2AResponse(
            request_id=request.request_id,
            status="completed",
            result_type="meal_recommendations",
            output=output,
            notes=[
                "minimal_a2a_boundary",
                "action=recommend_meal",
            ],
        )


@lru_cache(maxsize=1)
def get_a2a_service() -> A2AService:
    return A2AService(get_recommender_service())
