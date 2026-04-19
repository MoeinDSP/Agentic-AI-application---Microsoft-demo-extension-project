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


class ClusteringWorker(Worker[Context]):
    """
    Processes clustering tasks received via A2A protocol.

    Accepts a JSON payload with place_candidates, trip_start, and trip_end.
    Returns clustered_place_candidates as both a TextPart and DataPart artifact.
    """

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        assert task is not None

        await self.storage.update_task(task["id"], state="working")

        context: Context = await self.storage.load_context(task["context_id"]) or []
        context.extend(task.get("history", []))

        user_text = self._extract_user_text(task)

        try:
            print(f"\n📍 Clustering agent received: {user_text[:120]}...")

            response_text = await run_agent(user_text)
            result = self._parse_result(response_text)

            num_clusters = len(result.get("clustered_place_candidates", []))
            print(f"   ✅ Produced {num_clusters} cluster(s)")

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
                ),
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    name="clustered_places",
                    parts=[DataPart(data=result, kind="data")],
                ),
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

    def _parse_result(self, response_text: str) -> dict:
        """Parse the LLM's JSON response into a result dict."""
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:-1])
        return json.loads(clean)
