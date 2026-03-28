from typing import Annotated

from fastapi import APIRouter, Depends

from agent3.models.a2a import AgentCard
from agent3.models.plan import HealthResponse, PlanRequest, PlanResponse
from agent3.services.a2a import AgentCardService, get_agent_card_service
from agent3.services.planner import PlannerService, get_planner_service

router = APIRouter()
PlannerDependency = Annotated[PlannerService, Depends(get_planner_service)]
AgentCardDependency = Annotated[AgentCardService, Depends(get_agent_card_service)]


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/.well-known/agent-card.json",
    response_model=AgentCard,
    tags=["a2a"],
)
async def get_agent_card(
    agent_card_service: AgentCardDependency,
) -> AgentCard:
    return agent_card_service.build_agent_card()


@router.post("/v1/plan", response_model=PlanResponse, tags=["planning"])
async def plan_day(
    request: PlanRequest,
    planner: PlannerDependency,
) -> PlanResponse:
    return planner.build_plan(request)
