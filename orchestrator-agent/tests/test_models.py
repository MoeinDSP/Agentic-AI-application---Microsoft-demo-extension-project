"""Unit tests for orchestrator models."""
from __future__ import annotations

from datetime import datetime

import pytest

from app.models import (
    BudgetConstraint,
    DerivedParams,
    Location,
    TripRequest,
)


class TestDerivedParams:

    @pytest.mark.unit
    def test_basic_derivation(self):
        req = TripRequest(
            city="Milan",
            trip_start=datetime(2026, 6, 10, 9, 0),
            trip_end=datetime(2026, 6, 13, 18, 0),
            budget=BudgetConstraint(total_budget=1000.0),
        )
        derived = DerivedParams.from_request(req)

        assert derived.num_days == 4
        assert derived.activity_budget == 700.0
        assert derived.food_budget_total == 300.0
        assert derived.food_budget_per_day == 75.0

    @pytest.mark.unit
    def test_no_budget(self):
        req = TripRequest(
            city="Rome",
            trip_start=datetime(2026, 7, 1, 9, 0),
            trip_end=datetime(2026, 7, 3, 18, 0),
        )
        derived = DerivedParams.from_request(req)

        assert derived.num_days == 3
        assert derived.activity_budget is None
        assert derived.food_budget_per_day is None

    @pytest.mark.unit
    def test_single_day_trip(self):
        req = TripRequest(
            city="Amsterdam",
            trip_start=datetime(2026, 5, 20, 8, 0),
            trip_end=datetime(2026, 5, 20, 22, 0),
            budget=BudgetConstraint(total_budget=200.0),
        )
        derived = DerivedParams.from_request(req)

        assert derived.num_days == 1
        assert derived.food_budget_per_day == 60.0

    @pytest.mark.unit
    def test_trip_request_json_roundtrip(self):
        req = TripRequest(
            city="Paris",
            trip_start=datetime(2026, 8, 1, 9, 0),
            trip_end=datetime(2026, 8, 5, 18, 0),
            location=Location(latitude=48.8566, longitude=2.3522),
            budget=BudgetConstraint(total_budget=2000.0),
            preferences=["museums", "architecture"],
        )
        serialised = req.model_dump_json()
        deserialised = TripRequest.model_validate_json(serialised)
        assert deserialised == req
