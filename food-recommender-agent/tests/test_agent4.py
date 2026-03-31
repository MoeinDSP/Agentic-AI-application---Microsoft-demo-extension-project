from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from app.core.config import settings
from app.models import FoodRecommenderInput, FoodRecommenderOutput, Location, MealSlot
from app.tools.places_api import _budget_to_max_price


# ── Helpers ───────────────────────────────────────────────────────────────────

def agent4_is_running() -> bool:
    """Return True if Agent 4 is reachable (used to skip integration tests)."""
    try:
        resp = httpx.get(
            f"{settings.agent_url}/.well-known/agent.json",
            timeout=3,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def milan_duomo() -> Location:
    return Location(latitude=45.4654, longitude=9.1859)


@pytest.fixture
def lunch_input(milan_duomo: Location) -> FoodRecommenderInput:
    return FoodRecommenderInput(
        timeofday=MealSlot.lunch,
        searchcenter=milan_duomo,
        searchradiusmeters=800,
        budgetpermealperperson=25.0,
        preferences=["italian"],
    )


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestModels:
    @pytest.mark.unit
    def test_food_recommender_input_valid(self, lunch_input):
        assert lunch_input.timeofday == MealSlot.lunch
        assert lunch_input.searchradiusmeters == 800

    @pytest.mark.unit
    def test_food_recommender_input_defaults(self, milan_duomo):
        inp = FoodRecommenderInput(
            timeofday=MealSlot.dinner,
            searchcenter=milan_duomo,
        )
        assert inp.searchradiusmeters == 1000
        assert inp.budgetpermealperperson is None
        assert inp.preferences is None

    @pytest.mark.unit
    def test_food_recommender_input_radius_bounds(self, milan_duomo):
        with pytest.raises(Exception):
            FoodRecommenderInput(
                timeofday=MealSlot.lunch,
                searchcenter=milan_duomo,
                searchradiusmeters=50,  # below minimum (100)
            )

    @pytest.mark.unit
    def test_meal_slot_enum_values(self):
        assert MealSlot.breakfast == "breakfast"
        assert MealSlot.lunch    == "lunch"
        assert MealSlot.dinner   == "dinner"

    @pytest.mark.unit
    def test_food_recommender_output_empty(self):
        out = FoodRecommenderOutput(restaurantcandidates=[])
        assert out.restaurantcandidates == []

    @pytest.mark.unit
    def test_food_recommender_input_json_roundtrip(self, lunch_input):
        serialised   = lunch_input.model_dump_json()
        deserialised = FoodRecommenderInput.model_validate_json(serialised)
        assert deserialised == lunch_input


class TestBudgetMapping:
    @pytest.mark.unit
    def test_budget_below_15_maps_to_1(self):
        assert _budget_to_max_price(10.0) == 1

    @pytest.mark.unit
    def test_budget_below_35_maps_to_2(self):
        assert _budget_to_max_price(25.0) == 2

    @pytest.mark.unit
    def test_budget_below_70_maps_to_3(self):
        assert _budget_to_max_price(50.0) == 3

    @pytest.mark.unit
    def test_budget_above_70_returns_none(self):
        assert _budget_to_max_price(100.0) is None

    @pytest.mark.unit
    def test_budget_exact_threshold_15(self):
        # 15.0 is NOT < 15 → maps to level 2
        assert _budget_to_max_price(15.0) == 2


class TestWorkerPromptBuilding:
    @pytest.mark.unit
    def test_build_prompt_structured(self, lunch_input):
        from app.worker import FoodRecommenderWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = FoodRecommenderWorker(
            storage=InMemoryStorage(),
            broker=InMemoryBroker(),
        )
        prompt = worker._build_prompt(lunch_input.model_dump_json())
        assert "lunch" in prompt
        assert "45.4654" in prompt
        assert "25" in prompt
        assert "italian" in prompt

    @pytest.mark.unit
    def test_build_prompt_freeform_fallback(self):
        from app.worker import FoodRecommenderWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = FoodRecommenderWorker(
            storage=InMemoryStorage(),
            broker=InMemoryBroker(),
        )
        raw = "find me a nice pizza place near the station"
        assert worker._build_prompt(raw) == raw


class TestWorkerArtifacts:
    @pytest.mark.unit
    def test_make_artifacts_valid_json(self):
        from app.worker import FoodRecommenderWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = FoodRecommenderWorker(
            storage=InMemoryStorage(),
            broker=InMemoryBroker(),
        )
        response = json.dumps({"restaurantcandidates": [{"id": "1", "name": "Test"}]})
        artifacts = worker._make_artifacts(response)

        # Should produce TextPart artifact + DataPart artifact
        assert len(artifacts) == 2
        kinds = [p.get("kind") for a in artifacts for p in a.parts]
        assert "text" in kinds
        assert "data" in kinds

    @pytest.mark.unit
    def test_make_artifacts_invalid_json(self):
        from app.worker import FoodRecommenderWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = FoodRecommenderWorker(
            storage=InMemoryStorage(),
            broker=InMemoryBroker(),
        )
        artifacts = worker._make_artifacts("not valid json at all")
        # Only TextPart artifact — no crash
        assert len(artifacts) == 1

    @pytest.mark.unit
    def test_make_artifacts_strips_code_fences(self):
        from app.worker import FoodRecommenderWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = FoodRecommenderWorker(
            storage=InMemoryStorage(),
            broker=InMemoryBroker(),
        )
        fenced = '```json\n{"restaurantcandidates": []}\n```'
        artifacts = worker._make_artifacts(fenced)
        assert len(artifacts) == 2


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(
    not agent4_is_running(),
    reason="Agent 4 is not running at AGENT_URL",
)
class TestAgent4Integration:

    @pytest.mark.asyncio
    async def test_agent_card_is_reachable(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.agent_url}/.well-known/agent.json",
                timeout=5,
            )
        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "version" in card

    @pytest.mark.asyncio
    async def test_full_a2a_round_trip_lunch(self, lunch_input):
        """Send a lunch task and verify a completed response with candidates."""
        import asyncio

        task_id = str(uuid.uuid4())
        body = {
            "id": task_id,
            "message": {
                "role": "user",
                "kind": "message",
                "message_id": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": lunch_input.model_dump_json()}],
            },
        }

        async with httpx.AsyncClient() as client:
            # Send task
            send_resp = await client.post(
                f"{settings.agent_url}/tasks/send",
                json=body,
                timeout=15,
            )
            assert send_resp.status_code == 200

            # Poll until completed (max 60s)
            for _ in range(60):
                poll_resp = await client.get(
                    f"{settings.agent_url}/tasks/{task_id}",
                    timeout=10,
                )
                assert poll_resp.status_code == 200
                task = poll_resp.json()
                state = task.get("status", {}).get("state", "")

                if state == "completed":
                    break
                if state in ("canceled", "failed"):
                    pytest.fail(f"Task ended with state: {state}")

                await asyncio.sleep(1)
            else:
                pytest.fail("Task did not complete within 60s")

        # Verify at least one artifact was returned
        assert len(task.get("artifacts", [])) > 0

    @pytest.mark.asyncio
    async def test_task_cancel(self, milan_duomo):
        """Submit a task then immediately cancel it."""
        inp = FoodRecommenderInput(
            timeofday=MealSlot.breakfast,
            searchcenter=milan_duomo,
            searchradiusmeters=500,
        )
        task_id = str(uuid.uuid4())
        body = {
            "id": task_id,
            "message": {
                "role": "user",
                "kind": "message",
                "message_id": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": inp.model_dump_json()}],
            },
        }

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{settings.agent_url}/tasks/send",
                json=body,
                timeout=10,
            )
            cancel_resp = await client.post(
                f"{settings.agent_url}/tasks/{task_id}/cancel",
                timeout=10,
            )
        assert cancel_resp.status_code == 200