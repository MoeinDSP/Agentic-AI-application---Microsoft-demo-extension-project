"""Integration tests for the orchestrator agent."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

import httpx
import pytest

from app.core.config import settings
from app.models import BudgetConstraint, Location, TripRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def orchestrator_is_running() -> bool:
    try:
        resp = httpx.get(
            f"{settings.agent_url}/.well-known/agent.json",
            timeout=3,
        )
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture
def milan_request() -> TripRequest:
    return TripRequest(
        city="Milan",
        trip_start=datetime(2026, 6, 10, 9, 0),
        trip_end=datetime(2026, 6, 12, 22, 0),
        location=Location(latitude=45.4654, longitude=9.1859),
        budget=BudgetConstraint(total_budget=600.0),
        trip_reason="city break",
        preferences=["museums", "architecture"],
    )


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(
    not orchestrator_is_running(),
    reason="Orchestrator is not running at AGENT_URL",
)
class TestOrchestratorIntegration:

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
        assert card["name"] == settings.agent_name

    @pytest.mark.asyncio
    async def test_full_a2a_round_trip(self, milan_request):
        send_body = {
            "jsonrpc": "2.0",
            "id":      str(uuid.uuid4()),
            "method":  "message/send",
            "params": {
                "message": {
                    "role":       "user",
                    "kind":       "message",
                    "messageId": str(uuid.uuid4()),
                    "parts":      [{"kind": "text", "text": milan_request.model_dump_json()}],
                },
            },
        }

        async with httpx.AsyncClient() as client:
            send_resp = await client.post(
                f"{settings.agent_url}/",
                json=send_body,
                timeout=15,
            )
            assert send_resp.status_code == 200

            task_id = send_resp.json().get("result", {}).get("id")
            assert task_id is not None

            # Poll until completed (max 300s — orchestrator calls multiple agents)
            for _ in range(150):
                poll_body = {
                    "jsonrpc": "2.0",
                    "id":      str(uuid.uuid4()),
                    "method":  "tasks/get",
                    "params":  {"id": task_id},
                }
                poll_resp = await client.post(
                    f"{settings.agent_url}/",
                    json=poll_body,
                    timeout=10,
                )
                assert poll_resp.status_code == 200
                data  = poll_resp.json()
                task  = data.get("result", data)
                state = task.get("status", {}).get("state", "")

                if state == "completed":
                    break
                if state in ("canceled", "failed"):
                    pytest.fail(f"Task ended with state: {state}")

                await asyncio.sleep(2)
            else:
                pytest.fail("Task did not complete within 300s")

        assert len(task.get("artifacts", [])) > 0

    @pytest.mark.asyncio
    async def test_task_cancel(self, milan_request):
        send_body = {
            "jsonrpc": "2.0",
            "id":      str(uuid.uuid4()),
            "method":  "message/send",
            "params": {
                "message": {
                    "role":       "user",
                    "kind":       "message",
                    "messageId": str(uuid.uuid4()),
                    "parts":      [{"kind": "text", "text": milan_request.model_dump_json()}],
                },
            },
        }

        async with httpx.AsyncClient() as client:
            send_resp = await client.post(
                f"{settings.agent_url}/", json=send_body, timeout=10
            )
            task_id = send_resp.json().get("result", {}).get("id")
            assert task_id is not None

            cancel_body = {
                "jsonrpc": "2.0",
                "id":      str(uuid.uuid4()),
                "method":  "tasks/cancel",
                "params":  {"id": task_id},
            }
            cancel_resp = await client.post(
                f"{settings.agent_url}/", json=cancel_body, timeout=10
            )
        assert cancel_resp.status_code == 200
