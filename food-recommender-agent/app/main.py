from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fasta2a import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage

from app.core.config import settings
from app.worker import FoodRecommenderWorker


# ── Infrastructure ────────────────────────────────────────────────────────────

storage = InMemoryStorage()
broker  = InMemoryBroker()
worker  = FoodRecommenderWorker(storage=storage, broker=broker)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastA2A) -> AsyncIterator[None]:
    """
    Starts the task manager and worker on app startup,
    shuts them down cleanly on app shutdown.
    """
    async with app.task_manager:
        async with worker.run():
            yield


# ── FastA2A app ───────────────────────────────────────────────────────────────
# Automatically exposes:
#   GET  /.well-known/agent.json  →  A2A agent card (discovery)
#   POST /tasks/send              →  submit a new task
#   GET  /tasks/{id}              →  poll task state + result
#   POST /tasks/{id}/cancel       →  cancel a running task

app = FastA2A(
    storage=storage,
    broker=broker,
    lifespan=lifespan,
    name=settings.agent_name,
    description=settings.agent_description,
    url=settings.agent_url,
    version=settings.agent_version,
)