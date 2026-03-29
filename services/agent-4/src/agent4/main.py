from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent4.api.routes import router
from agent4.core.config import get_settings
from agent4.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger("agent4.startup", environment=settings.environment)
    logger.info("application_starting", extra={"event": "startup"})
    yield
    logger.info("application_stopping", extra={"event": "shutdown"})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Agent 4 deterministic mock meal recommendation service.",
        version=settings.service_version,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
