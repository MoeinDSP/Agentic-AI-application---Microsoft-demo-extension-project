from __future__ import annotations

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

storage = InMemoryStorage()
broker  = InMemoryBroker()
worker  = OrchestratorWorker(storage=storage, broker=broker)


@asynccontextmanager
async def lifespan(app: FastA2A) -> AsyncIterator[None]:
    try:
        async with app.task_manager:
            async with worker.run():
                print(f"✅ Agent 0 started — {settings.agent_name} v{settings.agent_version}")
                print(f"   Agent card: {settings.agent_url}/.well-known/agent.json")
                print(f"   Downstream: Agent1={settings.agent1_url}  Agent2={settings.agent2_url}  Agent3={settings.agent3_url}")
                yield
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


async def agent_card_handler(request: Request) -> Response:
    return JSONResponse({
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
    })


app = FastA2A(
    storage=storage,
    broker=broker,
    lifespan=lifespan,
    name=settings.agent_name,
    description=settings.agent_description,
    url=settings.agent_url,
    version=settings.agent_version,
)

app.routes.insert(
    0,
    Route("/.well-known/agent.json", endpoint=agent_card_handler, methods=["GET"]),
)
