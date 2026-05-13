from __future__ import annotations

import json
import uuid
from typing import Any

from fasta2a import Worker
from fasta2a.schema import (
    Artifact,
    DataPart,
    Message,
    TaskIdParams,
    TaskSendParams,
    TextPart,
)

from app.agent import run_agent

Context = list[Message]


class OrchestratorWorker(Worker[Context]):
    """
    Bridges FastA2A tasks to the LLM agent loop.
    The LLM decides which tools to call — there is no hardcoded pipeline.
    """

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        assert task is not None

        await self.storage.update_task(task["id"], state="working")

        context: Context = await self.storage.load_context(task["context_id"]) or []
        context.extend(task.get("history", []))

        user_text = self._extract_user_text(task)
        conversation_history = self._build_openai_history(context)

        try:
            print(f"\n🗺️  Orchestrator received: {user_text[:120]}...")

            # Run the LLM tool-calling loop
            response_text, updated_history = await run_agent(
                user_message=user_text,
                conversation_history=conversation_history,
            )

            print(f"   💬 Response: {response_text[:100]}...")

            # Build A2A message + artifacts
            agent_message = Message(
                role="agent",
                parts=[TextPart(text=response_text, kind="text")],
                kind="message",
                message_id=str(uuid.uuid4()),
            )

            context.append(agent_message)

            artifacts: list[Artifact] = [
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    name="agent_response",
                    parts=[TextPart(text=response_text, kind="text")],
                )
            ]

            await self.storage.update_context(task["context_id"], context)
            await self.storage.update_task(
                task["id"],
                state="completed",
                new_messages=[agent_message],
                new_artifacts=artifacts,
            )

        except Exception as e:
            import traceback
            print(f"\n❌ run_task FAILED — {type(e).__name__}: {e}")
            traceback.print_exc()
            await self.storage.update_task(task["id"], state="failed")

    async def cancel_task(self, params: TaskIdParams) -> None:
        task = await self.storage.load_task(params["id"])
        if task:
            await self.storage.update_task(params["id"], state="canceled")

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return history

    def build_artifacts(self, result: Any) -> list[Artifact]:
        return []

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_user_text(self, task: dict) -> str:
        for msg in reversed(task.get("history", [])):
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    return part.get("text", "")
        return ""

    def _build_openai_history(self, context: Context) -> list[dict]:
        """Convert A2A message history to OpenAI message format for multi-turn."""
        history = []
        for msg in context:
            role = "user" if msg.get("role") == "user" else "assistant"
            for part in msg.get("parts", []):
                if part.get("kind") == "text" and part.get("text"):
                    history.append({"role": role, "content": part["text"]})
        return history
