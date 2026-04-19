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
from app.models import ClusteringRequest
from app.services.clustering import cluster_places
from app.worker import ClusteringWorker

# ── Infrastructure ────────────────────────────────────────────────────────────

storage = InMemoryStorage()
broker  = InMemoryBroker()
worker  = ClusteringWorker(storage=storage, broker=broker)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastA2A) -> AsyncIterator[None]:
    try:
        async with app.task_manager:
            async with worker.run():
                print(f"✅ Agent 2 started — {settings.agent_name} v{settings.agent_version}")
                print(f"   Agent card  : {settings.agent_url}/.well-known/agent.json")
                print(f"   REST cluster: {settings.agent_url}/cluster")
                yield
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


# ── Agent card ────────────────────────────────────────────────────────────────

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


# ── REST /cluster endpoint (used by Agent 3 — Daily Scheduler) ────────────────

async def cluster_handler(request: Request) -> Response:
    """
    Synchronous REST endpoint for Agent 3 to call directly.
    Accepts ClusteringRequest JSON, returns ClusteringResponse JSON.
    """
    try:
        body = await request.json()
        req = ClusteringRequest.model_validate(body)
        clusters = cluster_places(req.place_candidates, req.trip_start, req.trip_end)
        return JSONResponse({
            "clustered_place_candidates": [
                [p.model_dump() for p in cluster]
                for cluster in clusters
            ]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


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

app.routes.insert(
    0,
    Route("/.well-known/agent.json", endpoint=agent_card_handler, methods=["GET"]),
)
app.routes.insert(
    1,
    Route("/cluster", endpoint=cluster_handler, methods=["POST"]),
)
