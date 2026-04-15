from __future__ import annotations

import json
import uuid
from datetime import timedelta
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

from app.core.config import settings
from app.models import DailySchedule, DerivedParams, PlaceCandidate, TripRequest
from app.agent import generate_itinerary_summary, parse_user_request
from app.services.a2a_client_helper import (
    call_clustering_agent,
    call_place_recommender,
)
from app.services.scheduler import schedule_day
from app.services.verification import verify_itinerary

# Type alias
Context = list[Message]


class OrchestratorWorker(Worker[Context]):
    """
    Executes trip-planning orchestration tasks.
    Coordinates Agent 1 → Agent 2 → Agent 3/4 → Verification
    and assembles the final itinerary.
    """

    # ── Main task handler ─────────────────────────────────────────────────────

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        assert task is not None

        await self.storage.update_task(task["id"], state="working")

        context: Context = await self.storage.load_context(task["context_id"]) or []
        context.extend(task.get("history", []))

        user_text = self._extract_user_text(task)

        try:
            request = await self._parse_request(user_text)
            if request is None:
                await self._fail_with_message(
                    task, context,
                    "Could not understand the trip request. "
                    "Please provide city, dates, and optionally a budget.",
                )
                return

            derived = DerivedParams.from_request(
                request,
                activity_ratio=settings.activity_budget_ratio,
                food_ratio=settings.food_budget_ratio,
            )

            print(f"\n🗺️  Orchestrator received trip request:")
            print(f"   City       : {request.city}")
            print(f"   Dates      : {request.trip_start} → {request.trip_end}")
            print(f"   Num days   : {derived.num_days}")
            print(f"   Budget     : {request.budget}")

            # ── Step 1 — Place Recommender (Agent 1 via A2A) ──────────────
            print("\n📍 Step 1 — Calling Agent 1 (Place Recommender)...")

            raw_places = await call_place_recommender(
                city=request.city,
                trip_start=request.trip_start.isoformat(),
                trip_end=request.trip_end.isoformat(),
                budget=derived.activity_budget,
                trip_reason=request.trip_reason,
                preferences=request.preferences,
            )

            print(f"   ✅ Received {len(raw_places)} place candidates")

            if not raw_places:
                result = self._build_result(request, derived, [], [
                    "Agent 1 returned no place candidates."
                ])
                await self._complete_task(task, context, result)
                return

            # Validate place candidates into Pydantic models
            place_candidates = self._parse_places(raw_places)
            print(f"   ✅ Validated {len(place_candidates)} places")

            # ── Step 2 — Clustering (Agent 2 via REST) ────────────────────
            print("\n🧩 Step 2 — Calling Agent 2 (Clustering)...")

            clustered_raw = await call_clustering_agent(
                place_candidates=[p.model_dump(mode="json") for p in place_candidates],
                trip_start=request.trip_start.isoformat(),
                trip_end=request.trip_end.isoformat(),
            )

            print(f"   ✅ Received {len(clustered_raw)} clusters")

            # ── Step 3 — Daily Scheduling (per cluster, calls Agent 4) ────
            print("\n📅 Step 3 — Scheduling each day...")

            daily_schedules: list[DailySchedule] = []

            for day_idx, cluster in enumerate(clustered_raw):
                day_date = (request.trip_start + timedelta(days=day_idx)).date()
                print(f"   Day {day_idx + 1} ({day_date}) — {len(cluster)} places")

                # Parse the cluster back into PlaceCandidate models
                cluster_places = self._parse_places(cluster)

                schedule = await schedule_day(
                    places=cluster_places,
                    day_date=day_date,
                    day_start_hour=9,
                    day_end_hour=22,
                    food_budget_per_day=derived.food_budget_per_day,
                    preferences=request.preferences,
                )

                daily_schedules.append(schedule)
                print(f"   ✅ Day {day_idx + 1}: {len(schedule.events)} events scheduled")

            # ── Step 4 — Verification ─────────────────────────────────────
            print("\n✔️  Step 4 — Verifying itinerary...")

            warnings = verify_itinerary(
                daily_schedules=daily_schedules,
                activity_budget=derived.activity_budget,
                food_budget_total=derived.food_budget_total,
            )

            if warnings:
                for w in warnings:
                    print(f"   ⚠️  {w}")
            else:
                print("   ✅ All checks passed")

            # ── Build & return result ─────────────────────────────────────
            result = self._build_result(request, derived, daily_schedules, warnings)

            # ── Step 5 — LLM summary (non-blocking) ──────────────────────
            print("\n💬 Step 5 — Generating human-friendly summary...")
            try:
                summary = await generate_itinerary_summary(
                    json.dumps(result, ensure_ascii=False, default=str)
                )
                if summary:
                    result["summary"] = summary
                    print(f"   ✅ Summary: {summary[:80]}...")
            except Exception as e:
                print(f"   ⚠️  Summary generation skipped: {e}")

            await self._complete_task(task, context, result)

            print(f"\n🎉 Orchestrator completed — "
                  f"{len(daily_schedules)} days, "
                  f"{sum(len(d.events) for d in daily_schedules)} total events\n")

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

    async def _parse_request(self, user_text: str) -> TripRequest | None:
        """
        Two-step parsing:
          1. Try direct JSON → TripRequest (structured clients)
          2. Fall back to LLM extraction (free-text / natural language)
        """
        # Step 1 — Try structured JSON
        try:
            return TripRequest.model_validate_json(user_text)
        except Exception:
            pass

        # Step 2 — LLM extraction via OpenRouter
        print("   ℹ️  Input is not structured JSON — using LLM to parse...")
        parsed = await parse_user_request(user_text)
        if parsed is None:
            return None

        try:
            return TripRequest.model_validate(parsed)
        except Exception as e:
            print(f"   ⚠️  LLM output didn't match TripRequest schema: {e}")
            return None

    async def _fail_with_message(
        self, task: dict, context: Context, error_msg: str,
    ) -> None:
        """Complete the task with an error message (not a crash)."""
        result = {"error": error_msg, "daily_schedules": [], "warnings": [error_msg]}
        response_text = json.dumps(result, ensure_ascii=False)

        agent_message = Message(
            role="agent",
            parts=[TextPart(text=response_text, kind="text")],
            kind="message",
            message_id=str(uuid.uuid4()),
        )

        context.append(agent_message)
        await self.storage.update_context(task["context_id"], context)
        await self.storage.update_task(
            task["id"],
            state="completed",
            new_messages=[agent_message],
            new_artifacts=self._make_artifacts(response_text),
        )

    def _parse_places(self, raw_list: list[dict]) -> list[PlaceCandidate]:
        """Best-effort parse of place dicts into validated models."""
        places = []
        for item in raw_list:
            try:
                if isinstance(item, dict):
                    places.append(PlaceCandidate.model_validate(item))
                elif isinstance(item, PlaceCandidate):
                    places.append(item)
            except Exception as e:
                print(f"   ⚠️  Skipping invalid place: {e}")
        return places

    def _build_result(
        self,
        request: TripRequest,
        derived: DerivedParams,
        daily_schedules: list[DailySchedule],
        warnings: list[str],
    ) -> dict:
        return {
            "request": request.model_dump(mode="json"),
            "derived": derived.model_dump(mode="json"),
            "daily_schedules": [
                s.model_dump(mode="json") for s in daily_schedules
            ],
            "warnings": warnings,
        }

    async def _complete_task(
        self,
        task: dict,
        context: Context,
        result: dict,
    ) -> None:
        """Serialize result, build artifacts, update storage."""
        response_text = json.dumps(result, ensure_ascii=False, default=str)

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

    def _make_artifacts(self, response_text: str) -> list[Artifact]:
        artifacts: list[Artifact] = [
            Artifact(
                artifact_id=str(uuid.uuid4()),
                name="agent_response",
                parts=[TextPart(text=response_text, kind="text")],
            )
        ]

        try:
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:-1])

            data = json.loads(clean)

            if "daily_schedules" in data:
                artifacts.append(
                    Artifact(
                        artifact_id=str(uuid.uuid4()),
                        name="trip_itinerary",
                        parts=[DataPart(data=data, kind="data")],
                    )
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return artifacts
