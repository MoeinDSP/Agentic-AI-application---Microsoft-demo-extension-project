from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent3_mcp.api.routes import router
from agent3_mcp.core.config import get_settings
from agent3_mcp.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger("agent3_mcp.startup", environment=settings.environment)
    logger.info("application_starting", extra={"event": "startup"})
    yield
    logger.info("application_stopping", extra={"event": "shutdown"})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
