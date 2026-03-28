from typing import Annotated

from fastapi import APIRouter, Depends

from agent3.models.a2a import A2ARequest, A2AResponse, AgentCard
from agent3.models.plan import HealthResponse, PlanRequest, PlanResponse
from agent3.services.a2a import (
    A2AService,
    AgentCardService,
    get_a2a_service,
    get_agent_card_service,
)
from agent3.services.planner import PlannerService, get_planner_service

router = APIRouter()
PlannerDependency = Annotated[PlannerService, Depends(get_planner_service)]
AgentCardDependency = Annotated[AgentCardService, Depends(get_agent_card_service)]
A2ADependency = Annotated[A2AService, Depends(get_a2a_service)]


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


@router.post("/a2a", response_model=A2AResponse, tags=["a2a"])
async def handle_a2a_request(
    request: A2ARequest,
    a2a_service: A2ADependency,
) -> A2AResponse:
    return a2a_service.handle_request(request)


@router.post("/v1/plan", response_model=PlanResponse, tags=["planning"])
async def plan_day(
    request: PlanRequest,
    planner: PlannerDependency,
) -> PlanResponse:
    return planner.plan_day(request)
