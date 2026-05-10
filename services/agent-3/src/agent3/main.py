from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fasta2a import FastA2A
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent3.core.config import Settings, get_settings
from agent3.core.logging import configure_logging, get_logger
from agent3.services.a2a import build_a2a_runtime, build_agent_skill
from agent3.services.planner import PlannerService, get_planner_service


async def healthcheck(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def create_app(
    *,
    settings: Settings | None = None,
    planner: PlannerService | None = None,
) -> FastA2A:
    resolved_settings = settings or get_settings()
    resolved_planner = planner or get_planner_service()
    storage, broker, worker = build_a2a_runtime(resolved_planner)

    @asynccontextmanager
    async def lifespan(app: FastA2A) -> AsyncIterator[None]:
        configure_logging(resolved_settings)
        logger = get_logger("agent3.startup", environment=resolved_settings.environment)
        logger.info("application_starting", extra={"event": "startup"})
        async with app.task_manager:
            async with worker.run():
                yield
        logger.info("application_stopping", extra={"event": "shutdown"})

    return FastA2A(
        storage=storage,
        broker=broker,
        name="Daily Scheduler Agent",
        url=resolved_settings.public_base_url.rstrip("/"),
        version=resolved_settings.service_version,
        description=(
            "Plans a day itinerary from candidate places, routing constraints, "
            "and meal preferences."
        ),
        skills=[build_agent_skill()],
        routes=[Route("/health", endpoint=healthcheck, methods=["GET"])],
        lifespan=lifespan,
    )


app = create_app()
