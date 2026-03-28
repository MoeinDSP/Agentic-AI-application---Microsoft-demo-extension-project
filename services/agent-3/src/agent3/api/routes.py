from typing import Annotated

from fastapi import APIRouter, Depends

from agent3.models.plan import HealthResponse, PlanRequest, PlanResponse
from agent3.services.planner import PlannerService, get_planner_service

router = APIRouter()
PlannerDependency = Annotated[PlannerService, Depends(get_planner_service)]


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/v1/plan", response_model=PlanResponse, tags=["planning"])
async def plan_day(
    request: PlanRequest,
    planner: PlannerDependency,
) -> PlanResponse:
    return planner.build_plan(request)
