from functools import lru_cache

from agent3.core.config import Settings, get_settings
from agent3.models.a2a import (
    A2ARequest,
    A2AResponse,
    AgentAuth,
    AgentCard,
    AgentEndpoint,
    AgentSkill,
)
from agent3.services.planner import PlannerService, get_planner_service


class AgentCardService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_agent_card(self) -> AgentCard:
        return AgentCard(
            name="agent-3",
            description=(
                "Local-first itinerary planning agent with deterministic MVP planning output."
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
                    path="/v1/plan",
                    method="POST",
                    description="Direct planning API contract.",
                ),
                AgentEndpoint(
                    path=self._settings.a2a_path,
                    method="POST",
                    description="Minimal A2A-shaped planning request endpoint.",
                ),
            ],
            skills=[
                AgentSkill(
                    id="itinerary-planning",
                    name="Itinerary Planning",
                    description="Builds a deterministic placeholder day plan from candidate stops.",
                    tags=["planning", "itinerary", "local-first"],
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
                "Planner behavior remains deterministic placeholder logic in this MVP stage.",
            ],
        )


@lru_cache(maxsize=1)
def get_agent_card_service() -> AgentCardService:
    return AgentCardService(get_settings())


class A2AService:
    def __init__(self, planner: PlannerService) -> None:
        self._planner = planner

    def handle_request(self, request: A2ARequest) -> A2AResponse:
        output = self._planner.build_plan(request.input)
        return A2AResponse(
            request_id=request.request_id,
            status="completed",
            result_type="plan",
            output=output,
            notes=[
                "minimal_a2a_boundary",
                "action=plan_day",
            ],
        )


@lru_cache(maxsize=1)
def get_a2a_service() -> A2AService:
    return A2AService(get_planner_service())
