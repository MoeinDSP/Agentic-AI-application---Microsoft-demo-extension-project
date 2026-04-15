from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from fasta2a import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage

from app.core.config import settings
from app.worker import OrchestratorWorker

# ── Infrastructure ────────────────────────────────────────────────────────────

storage = InMemoryStorage()
broker  = InMemoryBroker()
worker  = OrchestratorWorker(storage=storage, broker=broker)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastA2A) -> AsyncIterator[None]:
    """
    Starts the task manager and worker on app startup.
    """
    try:
        async with app.task_manager:
            async with worker.run():
                print(f"✅ Agent 0 started — {settings.agent_name} v{settings.agent_version}")
                print(f"   Agent card: {settings.agent_url}/.well-known/agent.json")
                yield
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


# ── Manual agent card (guaranteed to work regardless of fasta2a version) ──────

async def agent_card_handler(request: Request) -> Response:
    card = {
        "name":        settings.agent_name,
        "description": settings.agent_description,
        "url":         settings.agent_url,
        "version":     settings.agent_version,
        "protocolVersion": "0.3.0",
        "capabilities": {
            "streaming":              False,
            "pushNotifications":      False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes":  ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
    }
    return JSONResponse(card)


# ── FastA2A app ───────────────────────────────────────────────────────────────

app = FastA2A(
    storage=storage,
    broker=broker,
    lifespan=lifespan,
    name=settings.agent_name,
    description=settings.agent_description,
    url=settings.agent_url,
    version=settings.agent_version,
)

# Override / guarantee the agent card route regardless of fasta2a internals
app.routes.insert(
    0,
    Route("/.well-known/agent.json", endpoint=agent_card_handler, methods=["GET"]),
)
