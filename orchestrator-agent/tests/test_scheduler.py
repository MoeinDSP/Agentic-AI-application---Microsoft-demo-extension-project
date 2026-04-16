"""Unit tests for the day scheduler service."""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from app.models import Location, PlaceCandidate
from app.services.scheduler import (
    estimate_travel_minutes,
    haversine_km,
    order_places_greedy,
    schedule_day,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def milan_places() -> list[PlaceCandidate]:
    """Three places in Milan, roughly 1-2km apart."""
    return [
        PlaceCandidate(
            id="duomo",
            name="Duomo di Milano",
            location=Location(latitude=45.4641, longitude=9.1919),
            estimated_visit_duration_minutes=90,
            category="cathedral",
            rating=4.8,
            priority_score=1.0,
        ),
        PlaceCandidate(
            id="castello",
            name="Castello Sforzesco",
            location=Location(latitude=45.4705, longitude=9.1794),
            estimated_visit_duration_minutes=75,
            category="castle",
            rating=4.5,
            priority_score=0.8,
        ),
        PlaceCandidate(
            id="navigli",
            name="Navigli District",
            location=Location(latitude=45.4494, longitude=9.1724),
            estimated_visit_duration_minutes=60,
            category="neighbourhood",
            rating=4.3,
            priority_score=0.6,
        ),
    ]


@pytest.fixture
def single_place() -> list[PlaceCandidate]:
    return [
        PlaceCandidate(
            id="p1",
            name="Only Place",
            location=Location(latitude=45.46, longitude=9.19),
            estimated_visit_duration_minutes=120,
        ),
    ]


# ── Geo helpers ───────────────────────────────────────────────────────────────

class TestGeoHelpers:
    @pytest.mark.unit
    def test_haversine_same_point(self):
        assert haversine_km(45.0, 9.0, 45.0, 9.0) == 0.0

    @pytest.mark.unit
    def test_haversine_known_distance(self):
        # Duomo to Castello ≈ 1.2 km
        dist = haversine_km(45.4641, 9.1919, 45.4705, 9.1794)
        assert 0.8 < dist < 1.6

    @pytest.mark.unit
    def test_travel_estimate_positive(self, milan_places):
        minutes = estimate_travel_minutes(milan_places[0], milan_places[1])
        assert minutes >= 5


# ── Ordering ──────────────────────────────────────────────────────────────────

class TestOrdering:
    @pytest.mark.unit
    def test_greedy_order_starts_with_highest_priority(self, milan_places):
        ordered = order_places_greedy(milan_places)
        assert ordered[0].id == "duomo"  # highest priority_score

    @pytest.mark.unit
    def test_greedy_order_preserves_all_places(self, milan_places):
        ordered = order_places_greedy(milan_places)
        assert len(ordered) == len(milan_places)
        assert set(p.id for p in ordered) == set(p.id for p in milan_places)

    @pytest.mark.unit
    def test_single_place_ordering(self, single_place):
        ordered = order_places_greedy(single_place)
        assert len(ordered) == 1

    @pytest.mark.unit
    def test_empty_list(self):
        ordered = order_places_greedy([])
        assert ordered == []


# ── Full scheduler ────────────────────────────────────────────────────────────

class TestScheduleDay:
    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    async def test_schedule_creates_events(self, mock_food, milan_places):
        # Agent 4 returns a fake restaurant
        mock_food.return_value = [{
            "id": "r1",
            "name": "Trattoria Test",
            "location": {"latitude": 45.465, "longitude": 9.185},
            "price_level": 2,
            "cuisines": ["Italian"],
            "rating": 4.2,
            "summary": None,
        }]

        schedule = await schedule_day(
            places=milan_places,
            day_date=date(2026, 6, 10),
            day_start_hour=9,
            day_end_hour=22,
            food_budget_per_day=60.0,
            preferences=["italian"],
        )

        assert schedule.date == date(2026, 6, 10)
        assert len(schedule.events) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    async def test_schedule_inserts_meals(self, mock_food, milan_places):
        mock_food.return_value = [{
            "id": "r1",
            "name": "Test Ristorante",
            "location": {"latitude": 45.465, "longitude": 9.185},
            "rating": 4.0,
        }]

        schedule = await schedule_day(
            places=milan_places,
            day_date=date(2026, 6, 10),
            day_start_hour=9,
            day_end_hour=22,
        )

        from app.models import MealEvent
        meal_events = [e for e in schedule.events if isinstance(e, MealEvent)]
        # Should have at least lunch inserted
        assert len(meal_events) >= 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    async def test_schedule_handles_agent4_failure(self, mock_food, milan_places):
        mock_food.side_effect = Exception("Agent 4 is down")

        # Should not raise — scheduler continues without meals
        schedule = await schedule_day(
            places=milan_places,
            day_date=date(2026, 6, 10),
        )

        assert schedule.date == date(2026, 6, 10)
        assert len(schedule.events) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    async def test_schedule_single_place(self, mock_food, single_place):
        mock_food.return_value = []

        schedule = await schedule_day(
            places=single_place,
            day_date=date(2026, 6, 10),
        )

        from app.models import VisitEvent
        visit_events = [e for e in schedule.events if isinstance(e, VisitEvent)]
        assert len(visit_events) == 1
        assert visit_events[0].place.id == "p1"

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.services.scheduler.call_food_recommender", new_callable=AsyncMock)
    async def test_schedule_empty_places(self, mock_food):
        mock_food.return_value = []

        schedule = await schedule_day(
            places=[],
            day_date=date(2026, 6, 10),
        )

        assert schedule.events == []
