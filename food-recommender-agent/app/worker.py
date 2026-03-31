from __future__ import annotations

import json
import uuid
from typing import Any

from google.genai.types import Content, Part

from fasta2a import Worker
from fasta2a.schema import (
    Artifact,
    DataPart,
    Message,
    TaskIdParams,
    TaskSendParams,
    TextPart,
)

from app.agent import runner, session_service
from app.models import FoodRecommenderInput

# Type alias — conversation context stored in FastA2A Storage
Context = list[Message]


class FoodRecommenderWorker(Worker[Context]):
    """
    Executes food-recommendation tasks by forwarding them to the
    Vertex AI ADK agent and writing the result back into the A2A task.
    """

    # ── Main task handler ─────────────────────────────────────────────────────

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        assert task is not None

        await self.storage.update_task(task["id"], state="working")

        context: Context = await self.storage.load_context(task["context_id"]) or []
        context.extend(task.get("history", []))

        user_text = self._extract_user_text(task)
        prompt    = self._build_prompt(user_text)
        session_id = task["context_id"]

        try:
            await self._ensure_session(session_id)
            response_text = await self._run_agent(session_id, prompt)

            agent_message = Message(
                role="agent",
                parts=[TextPart(text=response_text, kind="text")],
                kind="message",
                message_id=str(uuid.uuid4()),
            )

            context.append(agent_message)
            artifacts = self._make_artifacts(response_text)

            await self.storage.update_context(task["context_id"], context)
            await self.storage.update_task(
                task["id"],
                state="completed",
                new_messages=[agent_message],
                new_artifacts=artifacts,
            )

        except Exception as e:
            import traceback
            print(f"\n❌ run_task FAILED — task_id={task['id']}")
            print(f"   Error: {type(e).__name__}: {e}")
            traceback.print_exc()
            await self.storage.update_task(task["id"], state="failed")

    # ── Cancel handler ────────────────────────────────────────────────────────

    async def cancel_task(self, params: TaskIdParams) -> None:
        task = await self.storage.load_task(params["id"])
        if task:
            await self.storage.update_task(params["id"], state="canceled")

    # ── Required Worker overrides ─────────────────────────────────────────────

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return history

    def build_artifacts(self, result: Any) -> list[Artifact]:
        return self._make_artifacts(str(result))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_user_text(self, task: dict) -> str:
        for msg in reversed(task.get("history", [])):
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    return part.get("text", "")
        return ""

    def _build_prompt(self, user_text: str) -> str:
        """
        Try to parse user_text as FoodRecommenderInput and build a
        structured prompt. Fall back to raw text if parsing fails.
        """
        try:
            inp = FoodRecommenderInput.model_validate_json(user_text)
            return (
                f"Find restaurants for a {inp.timeofday.value} meal.\n"
                f"Location : lat={inp.searchcenter.latitude}, "
                f"lon={inp.searchcenter.longitude}\n"
                f"Radius   : {inp.searchradiusmeters}m\n"
                f"Budget   : {inp.budgetpermealperperson or 'unspecified'} EUR "
                f"per person\n"
                f"Preferences: {', '.join(inp.preferences or []) or 'none'}\n\n"
                "Return the result as a JSON object with key "
                "'restaurantcandidates'."
            )
        except Exception:
            # Accept free-form text queries (useful for manual testing)
            return user_text

    async def _ensure_session(self, session_id: str) -> None:
        session = await session_service.get_session(
            app_name="food_recommender",
            user_id="a2a_client",
            session_id=session_id,
        )
        if session is None:
            await session_service.create_session(
                app_name="food_recommender",
                user_id="a2a_client",
                session_id=session_id,
            )

    async def _run_agent(self, session_id: str, prompt: str) -> str:
        """Run the ADK agent and return the final response text."""
        async for event in runner.run_async(
            user_id="a2a_client",
            session_id=session_id,
            new_message=Content(parts=[Part(text=prompt)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                return event.content.parts[0].text or ""
        return ""

    def _make_artifacts(self, response_text: str) -> list[Artifact]:
        """
        Always emit a raw TextPart artifact.
        If the response contains valid JSON with 'restaurantcandidates',
        also emit a structured DataPart artifact for easy parsing by Agent 3.
        """
        artifacts: list[Artifact] = [
            Artifact(
                artifact_id=str(uuid.uuid4()),
                name="agent_response",
                parts=[TextPart(text=response_text, kind="text")],
            )
        ]

        try:
            # Strip markdown code fences if Gemini wraps output in ```json ... ```
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:-1])

            data = json.loads(clean)

            if "restaurantcandidates" in data:
                artifacts.append(
                    Artifact(
                        artifact_id=str(uuid.uuid4()),
                        name="restaurant_candidates",
                        parts=[DataPart(data=data, kind="data")],
                    )
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return artifacts