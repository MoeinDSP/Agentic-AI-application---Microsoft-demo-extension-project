from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from agent4.models.a2a import A2ARequest, A2AResponse, AgentCard
from agent4.models.meal import (
    HealthResponse,
    MealRecommendationRequest,
    MealRecommendationResponse,
)
from agent4.services.a2a import (
    A2AService,
    AgentCardService,
    get_a2a_service,
    get_agent_card_service,
)
from agent4.services.recommender import RecommenderService, get_recommender_service

router = APIRouter()
RecommenderDependency = Annotated[RecommenderService, Depends(get_recommender_service)]
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
    try:
        return a2a_service.handle_request(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/v1/recommend-meal",
    response_model=MealRecommendationResponse,
    tags=["recommendations"],
)
async def recommend_meal(
    request: MealRecommendationRequest,
    recommender: RecommenderDependency,
) -> MealRecommendationResponse:
    return recommender.recommend(request)
