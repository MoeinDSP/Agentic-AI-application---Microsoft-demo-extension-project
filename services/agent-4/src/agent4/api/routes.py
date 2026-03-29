from typing import Annotated

from fastapi import APIRouter, Depends

from agent4.models.meal import (
    HealthResponse,
    MealRecommendationRequest,
    MealRecommendationResponse,
)
from agent4.services.recommender import RecommenderService, get_recommender_service

router = APIRouter()
RecommenderDependency = Annotated[RecommenderService, Depends(get_recommender_service)]


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


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
