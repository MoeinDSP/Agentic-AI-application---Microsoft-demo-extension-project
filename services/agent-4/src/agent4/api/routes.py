from typing import Annotated

from fastapi import APIRouter, Depends

from agent4.models.a2a import AgentCard
from agent4.models.meal import (
    HealthResponse,
    MealRecommendationRequest,
    MealRecommendationResponse,
)
from agent4.services.a2a import AgentCardService, get_agent_card_service
from agent4.services.recommender import RecommenderService, get_recommender_service

router = APIRouter()
RecommenderDependency = Annotated[RecommenderService, Depends(get_recommender_service)]
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
