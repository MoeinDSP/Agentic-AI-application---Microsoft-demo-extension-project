from typing import Annotated

from fastapi import APIRouter, Depends

from agent3_mcp.models.tools import (
    HealthResponse,
    PlaceDetailsRequest,
    PlaceDetailsResponse,
    RouteEstimateRequest,
    RouteEstimateResponse,
)
from agent3_mcp.services.tools import ToolService, get_tool_service

router = APIRouter()
ToolDependency = Annotated[ToolService, Depends(get_tool_service)]


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/v1/tools/route-estimate",
    response_model=RouteEstimateResponse,
    tags=["tools"],
)
async def route_estimate(
    request: RouteEstimateRequest,
    tools: ToolDependency,
) -> RouteEstimateResponse:
    return tools.estimate_route(request)


@router.post(
    "/v1/tools/place-details",
    response_model=PlaceDetailsResponse,
    tags=["tools"],
)
async def place_details(
    request: PlaceDetailsRequest,
    tools: ToolDependency,
) -> PlaceDetailsResponse:
    return tools.get_place_details(request)
