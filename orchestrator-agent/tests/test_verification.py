"""Unit tests for the itinerary verification service."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from app.models import (
    DailySchedule,
    Location,
    MealEvent,
    MealSlot,
    PlaceCandidate,
    RestaurantCandidate,
    VisitEvent,
)
from app.services.verification import verify_itinerary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _place(name: str, lat: float = 45.46, lon: float = 9.19) -> PlaceCandidate:
    return PlaceCandidate(
        id=name.lower().replace(" ", "_"),
        name=name,
        location=Location(latitude=lat, longitude=lon),
        estimated_visit_duration_minutes=60,
        estimated_cost=10.0,
    )


def _restaurant(name: str, lat: float = 45.46, lon: float = 9.19) -> RestaurantCandidate:
    return RestaurantCandidate(
        id=name.lower().replace(" ", "_"),
        name=name,
        location=Location(latitude=lat, longitude=lon),
        rating=4.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVerification:

    @pytest.mark.unit
    def test_valid_itinerary_no_warnings(self):
        schedule = DailySchedule(
            date=date(2026, 6, 10),
            events=[
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 9, 0),
                    end_time=datetime(2026, 6, 10, 10, 30),
                    place=_place("Duomo"),
                ),
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 11, 0),
                    end_time=datetime(2026, 6, 10, 12, 0),
                    place=_place("Castello"),
                ),
            ],
        )

        warnings = verify_itinerary([schedule])
        assert warnings == []

    @pytest.mark.unit
    def test_detects_overlap(self):
        schedule = DailySchedule(
            date=date(2026, 6, 10),
            events=[
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 9, 0),
                    end_time=datetime(2026, 6, 10, 11, 0),
                    place=_place("Duomo"),
                ),
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 10, 30),  # overlaps!
                    end_time=datetime(2026, 6, 10, 12, 0),
                    place=_place("Castello"),
                ),
            ],
        )

        warnings = verify_itinerary([schedule])
        assert any("overlap" in w.lower() for w in warnings)

    @pytest.mark.unit
    def test_detects_budget_overrun(self):
        schedule = DailySchedule(
            date=date(2026, 6, 10),
            events=[
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 9, 0),
                    end_time=datetime(2026, 6, 10, 11, 0),
                    place=PlaceCandidate(
                        id="expensive",
                        name="Expensive Place",
                        location=Location(latitude=45.46, longitude=9.19),
                        estimated_visit_duration_minutes=120,
                        estimated_cost=500.0,
                    ),
                ),
            ],
        )

        warnings = verify_itinerary([schedule], activity_budget=100.0)
        assert any("exceeds" in w.lower() for w in warnings)

    @pytest.mark.unit
    def test_detects_very_short_visit(self):
        schedule = DailySchedule(
            date=date(2026, 6, 10),
            events=[
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 9, 0),
                    end_time=datetime(2026, 6, 10, 9, 5),  # only 5 minutes
                    place=_place("Quick Stop"),
                ),
            ],
        )

        warnings = verify_itinerary([schedule])
        assert any("too short" in w.lower() for w in warnings)

    @pytest.mark.unit
    def test_detects_empty_day(self):
        schedule = DailySchedule(date=date(2026, 6, 10), events=[])
        warnings = verify_itinerary([schedule])
        assert any("no events" in w.lower() for w in warnings)

    @pytest.mark.unit
    def test_meal_events_handled(self):
        schedule = DailySchedule(
            date=date(2026, 6, 10),
            events=[
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 9, 0),
                    end_time=datetime(2026, 6, 10, 11, 0),
                    place=_place("Duomo"),
                ),
                MealEvent(
                    time=datetime(2026, 6, 10, 12, 0),
                    meal_slot=MealSlot.lunch,
                    restaurant=_restaurant("Trattoria"),
                ),
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 13, 30),
                    end_time=datetime(2026, 6, 10, 15, 0),
                    place=_place("Gallery"),
                ),
            ],
        )

        warnings = verify_itinerary([schedule])
        assert warnings == []

    @pytest.mark.unit
    def test_empty_itinerary(self):
        warnings = verify_itinerary([])
        assert warnings == []

    @pytest.mark.unit
    def test_budget_ok_when_under(self):
        schedule = DailySchedule(
            date=date(2026, 6, 10),
            events=[
                VisitEvent(
                    start_time=datetime(2026, 6, 10, 9, 0),
                    end_time=datetime(2026, 6, 10, 11, 0),
                    place=_place("Duomo"),
                ),
            ],
        )

        warnings = verify_itinerary([schedule], activity_budget=500.0)
        # 10 EUR cost < 500 EUR budget — no warning
        assert not any("exceeds" in w.lower() for w in warnings)
