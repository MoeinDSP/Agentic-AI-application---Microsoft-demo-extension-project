"""
End-to-end pipeline test with mocked downstream agents.

Tests the full orchestrator flow without needing Agent 1, 2, or 4 running.
All external calls are patched with realistic mock data for a 3-day Milan trip.

Run:
    pytest tests/test_pipeline_mocked.py -v
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage

from app.models import (
    BudgetConstraint,
    DailySchedule,
    DerivedParams,
    Location,
    MealEvent,
    MealSlot,
    PlaceCandidate,
    TripRequest,
    VisitEvent,
)
from app.worker import OrchestratorWorker


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK DATA — realistic responses from each agent
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_PLACES: list[dict] = [
    {
        "id": "duomo_di_milano",
        "name": "Duomo di Milano",
        "location": {"latitude": 45.4641, "longitude": 9.1919, "address": "Piazza del Duomo"},
        "estimated_visit_duration_minutes": 90,
        "estimated_cost": 15.0,
        "category": "cathedral",
        "rating": 4.8,
        "summary": "Gothic cathedral and Milan's most iconic landmark.",
        "priority_score": 1.0,
        "opening_hours": [],
    },
    {
        "id": "castello_sforzesco",
        "name": "Castello Sforzesco",
        "location": {"latitude": 45.4705, "longitude": 9.1794, "address": "Piazza Castello"},
        "estimated_visit_duration_minutes": 75,
        "estimated_cost": 10.0,
        "category": "castle",
        "rating": 4.5,
        "summary": "15th-century castle housing several museums.",
        "priority_score": 0.9,
        "opening_hours": [],
    },
    {
        "id": "pinacoteca_di_brera",
        "name": "Pinacoteca di Brera",
        "location": {"latitude": 45.4720, "longitude": 9.1879, "address": "Via Brera 28"},
        "estimated_visit_duration_minutes": 90,
        "estimated_cost": 15.0,
        "category": "art museum",
        "rating": 4.7,
        "summary": "One of Italy's foremost collections of Italian Renaissance art.",
        "priority_score": 0.85,
        "opening_hours": [],
    },
    {
        "id": "navigli_district",
        "name": "Navigli District",
        "location": {"latitude": 45.4494, "longitude": 9.1724, "address": "Naviglio Grande"},
        "estimated_visit_duration_minutes": 120,
        "estimated_cost": 0.0,
        "category": "neighbourhood",
        "rating": 4.3,
        "summary": "Lively canal district with bars, boutiques, and street art.",
        "priority_score": 0.7,
        "opening_hours": [],
    },
    {
        "id": "galleria_vittorio_emanuele",
        "name": "Galleria Vittorio Emanuele II",
        "location": {"latitude": 45.4658, "longitude": 9.1901, "address": "Piazza del Duomo"},
        "estimated_visit_duration_minutes": 45,
        "estimated_cost": 0.0,
        "category": "shopping arcade",
        "rating": 4.6,
        "summary": "Italy's oldest active shopping gallery with ornate glass ceiling.",
        "priority_score": 0.75,
        "opening_hours": [],
    },
    {
        "id": "santa_maria_delle_grazie",
        "name": "Santa Maria delle Grazie",
        "location": {"latitude": 45.4660, "longitude": 9.1711, "address": "Piazza di Santa Maria delle Grazie"},
        "estimated_visit_duration_minutes": 60,
        "estimated_cost": 15.0,
        "category": "church / museum",
        "rating": 4.9,
        "summary": "Home to Leonardo da Vinci's The Last Supper.",
        "priority_score": 0.95,
        "opening_hours": [],
    },
]

# Agent 2 returns 3 clusters (one per trip day)
MOCK_CLUSTERS: list[list[dict]] = [
    # Day 1: Duomo area (Duomo + Galleria)
    [MOCK_PLACES[0], MOCK_PLACES[4]],
    # Day 2: Culture (Castello + Brera + Last Supper)
    [MOCK_PLACES[1], MOCK_PLACES[2], MOCK_PLACES[5]],
    # Day 3: Navigli (casual day)
    [MOCK_PLACES[3]],
]

# Agent 4 returns restaurant candidates per meal
MOCK_RESTAURANTS_LUNCH: list[dict] = [
    {
        "id": "trattoria_milanese",
        "name": "Trattoria Milanese",
        "location": {"latitude": 45.4635, "longitude": 9.1880, "address": "Via Santa Marta 11"},
        "price_level": 2,
        "cuisines": ["Italian", "Milanese"],
        "rating": 4.4,
        "summary": "Traditional Milanese cuisine since 1933.",
    },
    {
        "id": "luini_panzerotti",
        "name": "Luini Panzerotti",
        "location": {"latitude": 45.4655, "longitude": 9.1905, "address": "Via Santa Radegonda 16"},
        "price_level": 1,
        "cuisines": ["Italian", "Street Food"],
        "rating": 4.3,
        "summary": "Famous fried panzerotti, a Milan institution.",
    },
]

MOCK_RESTAURANTS_DINNER: list[dict] = [
    {
        "id": "ristorante_berton",
        "name": "Ristorante Berton",
        "location": {"latitude": 45.4800, "longitude": 9.1900, "address": "Viale della Liberazione 13"},
        "price_level": 3,
        "cuisines": ["Italian", "Fine Dining"],
        "rating": 4.6,
        "summary": "Michelin-starred contemporary Italian.",
    },
]

MOCK_SUMMARY = (
    "Your 3-day Milan adventure kicks off at the stunning Duomo di Milano "
    "and the elegant Galleria. Day 2 is packed with culture — Castello Sforzesco, "
    "the Brera art gallery, and da Vinci's Last Supper. You'll wrap up with a "
    "relaxed stroll through the Navigli canals. Buon viaggio!"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def milan_trip_request() -> TripRequest:
    return TripRequest(
        city="Milan",
        trip_start=datetime(2026, 6, 10, 9, 0),
        trip_end=datetime(2026, 6, 12, 22, 0),
        location=Location(latitude=45.4654, longitude=9.1859),
        budget=BudgetConstraint(total_budget=600.0),
        trip_reason="city break",
        preferences=["museums", "architecture", "italian food"],
    )


@pytest.fixture
def worker():
    storage = InMemoryStorage()
    broker  = InMemoryBroker()
    return OrchestratorWorker(storage=storage, broker=broker)


def _food_recommender_side_effect(
    meal_slot: str, **kwargs
) -> list[dict]:
    """Return different restaurants depending on meal slot."""
    if meal_slot == "lunch":
        return MOCK_RESTAURANTS_LUNCH
    elif meal_slot == "dinner":
        return MOCK_RESTAURANTS_DINNER
    # breakfast — return lunch options as fallback
    return MOCK_RESTAURANTS_LUNCH


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: simulate a full run_task cycle through storage
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_pipeline_via_worker(
    worker: OrchestratorWorker,
    payload: str,
) -> dict:
    """
    Submit a task to InMemoryStorage, run the worker, return the final task.
    This simulates what FastA2A does without needing the HTTP server.
    """
    from fasta2a.schema import Message, TextPart

    # Create a context + task in storage (mimicking what FastA2A does)
    context_id = str(uuid.uuid4())
    task_id    = str(uuid.uuid4())

    message = Message(
        role="user",
        kind="message",
        message_id=str(uuid.uuid4()),
        parts=[TextPart(text=payload, kind="text")],
    )

    # Manually seed the task into InMemoryStorage
    task = await worker.storage.submit_task(context_id, message)
    real_task_id = task["id"]

    # Run the worker's run_task
    await worker.run_task({"id": real_task_id})

    # Load the completed task
    result_task = await worker.storage.load_task(real_task_id)
    return result_task


# ═══════════════════════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineMocked:
    """
    End-to-end pipeline tests with all downstream agents mocked.
    Each test patches call_place_recommender, call_clustering_agent,
    call_food_recommender, and generate_itinerary_summary.
    """

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.generate_itinerary_summary", new_callable=AsyncMock)
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    @patch("app.worker.call_clustering_agent", new_callable=AsyncMock)
    @patch("app.worker.call_place_recommender", new_callable=AsyncMock)
    async def test_full_pipeline_structured_json(
        self,
        mock_places,
        mock_clustering,
        mock_food,
        mock_summary,
        worker,
        milan_trip_request,
    ):
        """Full pipeline with a structured JSON TripRequest payload."""
        # ── Arrange ───────────────────────────────────────────────────────
        mock_places.return_value = MOCK_PLACES
        mock_clustering.return_value = MOCK_CLUSTERS
        mock_food.side_effect = _food_recommender_side_effect
        mock_summary.return_value = MOCK_SUMMARY

        payload = milan_trip_request.model_dump_json()

        # ── Act ───────────────────────────────────────────────────────────
        task = await _run_pipeline_via_worker(worker, payload)

        # ── Assert — task completed ───────────────────────────────────────
        assert task is not None
        assert task["status"]["state"] == "completed"

        # ── Assert — artifacts exist ──────────────────────────────────────
        assert len(task.get("artifacts", [])) >= 1

        # Extract the itinerary from artifacts
        itinerary = None
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "data":
                    itinerary = part["data"]
                    break
            if itinerary:
                break

        # If no DataPart, fall back to TextPart
        if itinerary is None:
            for artifact in task.get("artifacts", []):
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text":
                        itinerary = json.loads(part["text"])
                        break

        assert itinerary is not None, "No itinerary found in artifacts"

        # ── Assert — structure is correct ─────────────────────────────────
        assert "daily_schedules" in itinerary
        assert "derived" in itinerary
        assert "request" in itinerary
        assert "summary" in itinerary

        schedules = itinerary["daily_schedules"]
        assert len(schedules) == 3, f"Expected 3 days, got {len(schedules)}"

        # ── Assert — derived values are correct ───────────────────────────
        derived = itinerary["derived"]
        assert derived["num_days"] == 3  # ceil((12th 22:00 - 10th 09:00) / 24h)
        assert derived["activity_budget"] == 420.0   # 0.70 * 600
        assert derived["food_budget_total"] == 180.0  # 0.30 * 600

        # ── Assert — each day has events ──────────────────────────────────
        for i, day in enumerate(schedules):
            assert "date" in day
            assert "events" in day
            assert len(day["events"]) > 0, f"Day {i + 1} has no events"

        # ── Assert — meals were inserted ──────────────────────────────────
        all_events = []
        for day in schedules:
            all_events.extend(day["events"])

        meal_events = [e for e in all_events if "restaurant" in e]
        visit_events = [e for e in all_events if "place" in e]

        assert len(visit_events) > 0, "No visit events in schedule"
        assert len(meal_events) > 0, "No meal events — Agent 4 mock not called?"

        # ── Assert — downstream agents were called ────────────────────────
        mock_places.assert_called_once()
        mock_clustering.assert_called_once()
        assert mock_food.call_count >= 1, "Agent 4 (food) was never called"
        mock_summary.assert_called_once()

        # ── Assert — summary was generated ────────────────────────────────
        assert itinerary["summary"] == MOCK_SUMMARY

        print(f"\n✅ Full pipeline test passed:")
        print(f"   {len(schedules)} days scheduled")
        print(f"   {len(visit_events)} visit events")
        print(f"   {len(meal_events)} meal events")
        print(f"   Summary: {itinerary['summary'][:60]}...")

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.generate_itinerary_summary", new_callable=AsyncMock)
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    @patch("app.worker.call_clustering_agent", new_callable=AsyncMock)
    @patch("app.worker.call_place_recommender", new_callable=AsyncMock)
    @patch("app.worker.parse_user_request", new_callable=AsyncMock)
    async def test_full_pipeline_free_text(
        self,
        mock_llm_parse,
        mock_places,
        mock_clustering,
        mock_food,
        mock_summary,
        worker,
    ):
        """Full pipeline with a natural-language request parsed by the LLM."""
        # ── Arrange ───────────────────────────────────────────────────────
        # Simulate what the LLM would extract from free text
        mock_llm_parse.return_value = {
            "city": "Milan",
            "trip_start": "2026-06-10T09:00:00",
            "trip_end": "2026-06-12T22:00:00",
            "location": None,
            "budget": {"total_budget": 600.0, "currency": "EUR"},
            "trip_reason": "city break",
            "preferences": ["museums", "architecture", "italian food"],
        }
        mock_places.return_value = MOCK_PLACES
        mock_clustering.return_value = MOCK_CLUSTERS
        mock_food.side_effect = _food_recommender_side_effect
        mock_summary.return_value = MOCK_SUMMARY

        # Free-text input (NOT valid JSON)
        payload = (
            "Plan me a 3-day trip to Milan starting June 10th 2026. "
            "Budget is 600 euros. I love museums, architecture, and italian food."
        )

        # ── Act ───────────────────────────────────────────────────────────
        task = await _run_pipeline_via_worker(worker, payload)

        # ── Assert ────────────────────────────────────────────────────────
        assert task is not None
        assert task["status"]["state"] == "completed"

        # LLM parse should have been called (JSON parse would have failed)
        mock_llm_parse.assert_called_once()

        # Rest of pipeline should have run
        mock_places.assert_called_once()
        mock_clustering.assert_called_once()

        itinerary = json.loads(
            next(
                part["text"]
                for artifact in task.get("artifacts", [])
                for part in artifact.get("parts", [])
                if part.get("kind") == "text"
            )
        )
        assert len(itinerary["daily_schedules"]) == 3

        print("\n✅ Free-text pipeline test passed (LLM parsing mocked)")

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.generate_itinerary_summary", new_callable=AsyncMock)
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    @patch("app.worker.call_clustering_agent", new_callable=AsyncMock)
    @patch("app.worker.call_place_recommender", new_callable=AsyncMock)
    async def test_pipeline_no_places_returns_warning(
        self,
        mock_places,
        mock_clustering,
        mock_food,
        mock_summary,
        worker,
        milan_trip_request,
    ):
        """When Agent 1 returns no places, pipeline should complete with a warning."""
        mock_places.return_value = []
        mock_clustering.return_value = []
        mock_food.return_value = []
        mock_summary.return_value = ""

        payload = milan_trip_request.model_dump_json()
        task = await _run_pipeline_via_worker(worker, payload)

        assert task["status"]["state"] == "completed"

        itinerary = json.loads(
            next(
                part["text"]
                for artifact in task.get("artifacts", [])
                for part in artifact.get("parts", [])
                if part.get("kind") == "text"
            )
        )

        assert "Agent 1 returned no place candidates" in itinerary["warnings"][0]
        assert itinerary["daily_schedules"] == []

        # Agent 2 should NOT have been called
        mock_clustering.assert_not_called()

        print("\n✅ No-places warning test passed")

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.generate_itinerary_summary", new_callable=AsyncMock)
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    @patch("app.worker.call_clustering_agent", new_callable=AsyncMock)
    @patch("app.worker.call_place_recommender", new_callable=AsyncMock)
    async def test_pipeline_agent4_failure_graceful(
        self,
        mock_places,
        mock_clustering,
        mock_food,
        mock_summary,
        worker,
        milan_trip_request,
    ):
        """When Agent 4 (food) fails, the schedule should still be created."""
        mock_places.return_value = MOCK_PLACES
        mock_clustering.return_value = MOCK_CLUSTERS
        mock_food.side_effect = Exception("Agent 4 connection refused")
        mock_summary.return_value = "Trip summary without meals."

        payload = milan_trip_request.model_dump_json()
        task = await _run_pipeline_via_worker(worker, payload)

        assert task["status"]["state"] == "completed"

        itinerary = json.loads(
            next(
                part["text"]
                for artifact in task.get("artifacts", [])
                for part in artifact.get("parts", [])
                if part.get("kind") == "text"
            )
        )

        schedules = itinerary["daily_schedules"]
        assert len(schedules) == 3

        # Visit events should still exist (meals may be missing)
        all_events = []
        for day in schedules:
            all_events.extend(day["events"])

        visit_events = [e for e in all_events if "place" in e]
        assert len(visit_events) > 0, "Visits should exist even when Agent 4 fails"

        print(f"\n✅ Agent 4 failure test passed — {len(visit_events)} visits still scheduled")

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.generate_itinerary_summary", new_callable=AsyncMock)
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    @patch("app.worker.call_clustering_agent", new_callable=AsyncMock)
    @patch("app.worker.call_place_recommender", new_callable=AsyncMock)
    async def test_pipeline_no_budget(
        self,
        mock_places,
        mock_clustering,
        mock_food,
        mock_summary,
        worker,
    ):
        """Pipeline should work without a budget constraint."""
        request = TripRequest(
            city="Rome",
            trip_start=datetime(2026, 7, 1, 9, 0),
            trip_end=datetime(2026, 7, 2, 22, 0),
            preferences=["history"],
        )

        mock_places.return_value = [MOCK_PLACES[0], MOCK_PLACES[1]]
        mock_clustering.return_value = [[MOCK_PLACES[0], MOCK_PLACES[1]]]
        mock_food.side_effect = _food_recommender_side_effect
        mock_summary.return_value = "A lovely day in Rome."

        task = await _run_pipeline_via_worker(worker, request.model_dump_json())

        assert task["status"]["state"] == "completed"

        itinerary = json.loads(
            next(
                part["text"]
                for artifact in task.get("artifacts", [])
                for part in artifact.get("parts", [])
                if part.get("kind") == "text"
            )
        )

        derived = itinerary["derived"]
        assert derived["activity_budget"] is None
        assert derived["food_budget_per_day"] is None
        assert len(itinerary["daily_schedules"]) >= 1

        print("\n✅ No-budget pipeline test passed")

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.generate_itinerary_summary", new_callable=AsyncMock)
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    @patch("app.worker.call_clustering_agent", new_callable=AsyncMock)
    @patch("app.worker.call_place_recommender", new_callable=AsyncMock)
    async def test_pipeline_summary_failure_still_completes(
        self,
        mock_places,
        mock_clustering,
        mock_food,
        mock_summary,
        worker,
        milan_trip_request,
    ):
        """If LLM summary fails, pipeline should still complete without it."""
        mock_places.return_value = MOCK_PLACES
        mock_clustering.return_value = MOCK_CLUSTERS
        mock_food.side_effect = _food_recommender_side_effect
        mock_summary.side_effect = Exception("OpenRouter rate limit")

        payload = milan_trip_request.model_dump_json()
        task = await _run_pipeline_via_worker(worker, payload)

        assert task["status"]["state"] == "completed"

        itinerary = json.loads(
            next(
                part["text"]
                for artifact in task.get("artifacts", [])
                for part in artifact.get("parts", [])
                if part.get("kind") == "text"
            )
        )

        # Summary should be absent or empty — but itinerary is fine
        assert len(itinerary["daily_schedules"]) == 3
        assert "summary" not in itinerary or itinerary.get("summary") == ""

        print("\n✅ Summary-failure test passed — itinerary still delivered")


class TestMockDataIntegrity:
    """Validate that the mock data itself is well-formed."""

    @pytest.mark.unit
    def test_mock_places_are_valid(self):
        for p in MOCK_PLACES:
            place = PlaceCandidate.model_validate(p)
            assert place.name
            assert place.location.latitude != 0

    @pytest.mark.unit
    def test_mock_clusters_cover_all_places(self):
        clustered_ids = set()
        for cluster in MOCK_CLUSTERS:
            for place in cluster:
                clustered_ids.add(place["id"])

        place_ids = {p["id"] for p in MOCK_PLACES}
        assert clustered_ids == place_ids, (
            f"Clusters are missing: {place_ids - clustered_ids}"
        )

    @pytest.mark.unit
    def test_mock_clusters_match_trip_days(self):
        # 3-day trip → 3 clusters
        assert len(MOCK_CLUSTERS) == 3

    @pytest.mark.unit
    def test_mock_restaurants_have_required_fields(self):
        for r in MOCK_RESTAURANTS_LUNCH + MOCK_RESTAURANTS_DINNER:
            assert "id" in r
            assert "name" in r
            assert "location" in r
            assert "latitude" in r["location"]


class TestDerivedParamsWithMockRequest:
    """Verify budget splitting with the mock trip request."""

    @pytest.mark.unit
    def test_milan_budget_split(self, milan_trip_request):
        derived = DerivedParams.from_request(milan_trip_request)

        assert derived.activity_budget == 420.0     # 70% of 600
        assert derived.food_budget_total == 180.0    # 30% of 600
        assert derived.num_days == 3                 # ceil((12th 22:00 - 10th 09:00) / 24h)
        assert derived.food_budget_per_day == 60.0   # 180 / 3
