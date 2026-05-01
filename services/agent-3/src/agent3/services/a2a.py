import json
import uuid
from functools import lru_cache
from typing import Any

from fasta2a import Skill, Worker
from fasta2a.broker import Broker, InMemoryBroker
from fasta2a.schema import (
    Artifact,
    DataPart,
    Message,
    TaskIdParams,
    TaskSendParams,
    TextPart,
)
from fasta2a.storage import InMemoryStorage

from agent3.core.config import Settings, get_settings
from agent3.models.plan import DaySchedulingRequest, DaySchedulingResult
from agent3.services.planner import PlannerService, get_planner_service

Context = list[Message]


class PlannerA2AWorker(Worker[Context]):
    def __init__(
        self,
        *,
        broker: Broker,
        storage: InMemoryStorage[Context],
        planner: PlannerService,
    ) -> None:
        super().__init__(broker=broker, storage=storage)
        self._planner = planner

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        assert task is not None

        await self.storage.update_task(task["id"], state="working")

        request = self._extract_request(params["message"])
        result = self._planner.build_plan(request)
        artifacts = self.build_artifacts(result)
        completion_message = self._build_completion_message(result)

        context = await self.storage.load_context(task["context_id"]) or []
        context.extend(task.get("history", []))
        context.append(completion_message)
        await self.storage.update_context(task["context_id"], context)

        await self.storage.update_task(
            task["id"],
            state="completed",
            new_messages=[completion_message],
            new_artifacts=artifacts,
        )

    async def cancel_task(self, params: TaskIdParams) -> None:
        task = await self.storage.load_task(params["id"])
        if task is None:
            return

        await self.storage.update_task(task["id"], state="canceled")

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return history

    def build_artifacts(self, result: DaySchedulingResult) -> list[Artifact]:
        payload = result.model_dump(mode="json")
        return [
            Artifact(
                artifact_id=str(uuid.uuid4()),
                name="day_schedule",
                description="Structured daily schedule result.",
                parts=[
                    DataPart(
                        kind="data",
                        data=payload,
                    )
                ],
            )
        ]

    def _extract_request(self, message: Message) -> DaySchedulingRequest:
        for part in message["parts"]:
            if "data" in part:
                return DaySchedulingRequest.model_validate(part["data"])
            if "text" in part:
                return DaySchedulingRequest.model_validate(json.loads(part["text"]))
        raise ValueError("message must include a JSON scheduling request part")

    def _build_completion_message(self, result: DaySchedulingResult) -> Message:
        scheduled_event_count = len(result.day_schedule.events)
        return Message(
            role="agent",
            kind="message",
            message_id=str(uuid.uuid4()),
            parts=[
                TextPart(
                    kind="text",
                    text=(
                        f"Generated a schedule with {scheduled_event_count} events and "
                        f"{len(result.unscheduled_places)} unscheduled places."
                    ),
                )
            ],
        )


def build_agent_skill() -> Skill:
    return Skill(
        id="day-scheduling",
        name="Day Scheduling",
        description=(
            "Builds a daily itinerary from candidate places, travel constraints, "
            "and meal preferences."
        ),
        tags=["planning", "itinerary", "travel"],
        examples=[
            "Build a one-day Rome itinerary with lunch and dinner.",
            "Schedule museum visits while minimizing travel between stops.",
        ],
        input_modes=["application/json"],
        output_modes=["application/json"],
    )


@lru_cache(maxsize=1)
def get_a2a_runtime() -> tuple[InMemoryStorage[Context], InMemoryBroker, PlannerA2AWorker]:
    planner = get_planner_service()
    storage: InMemoryStorage[Context] = InMemoryStorage()
    broker = InMemoryBroker()
    worker = PlannerA2AWorker(broker=broker, storage=storage, planner=planner)
    return storage, broker, worker


def build_a2a_runtime(
    planner: PlannerService | None = None,
) -> tuple[InMemoryStorage[Context], InMemoryBroker, PlannerA2AWorker]:
    runtime_planner = planner or get_planner_service()
    storage: InMemoryStorage[Context] = InMemoryStorage()
    broker = InMemoryBroker()
    worker = PlannerA2AWorker(broker=broker, storage=storage, planner=runtime_planner)
    return storage, broker, worker


@lru_cache(maxsize=1)
def get_agent_metadata() -> tuple[Settings, Skill]:
    settings = get_settings()
    return settings, build_agent_skill()
