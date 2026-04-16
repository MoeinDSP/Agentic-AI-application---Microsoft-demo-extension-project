"""
Orchestrator tests with mocked downstream agents and LLM.

Tests the tool-calling loop without needing any live agents or OpenRouter.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Message, TextPart
from fasta2a.storage import InMemoryStorage

from app.agent import _execute_tool, run_agent
from app.worker import OrchestratorWorker


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK DATA
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_PLACES_RESPONSE = json.dumps([
    {"id": "duomo", "name": "Duomo di Milano", "location": {"latitude": 45.4641, "longitude": 9.1919},
     "estimated_visit_duration_minutes": 90, "rating": 4.8, "priority_score": 1.0},
    {"id": "castello", "name": "Castello Sforzesco", "location": {"latitude": 45.4705, "longitude": 9.1794},
     "estimated_visit_duration_minutes": 75, "rating": 4.5, "priority_score": 0.9},
])

MOCK_CLUSTERS_RESPONSE = json.dumps({
    "clustered_place_candidates": [
        [{"id": "duomo", "name": "Duomo di Milano", "location": {"latitude": 45.4641, "longitude": 9.1919},
          "estimated_visit_duration_minutes": 90, "rating": 4.8}],
        [{"id": "castello", "name": "Castello Sforzesco", "location": {"latitude": 45.4705, "longitude": 9.1794},
          "estimated_visit_duration_minutes": 75, "rating": 4.5}],
    ]
})

MOCK_SCHEDULE_RESPONSE = json.dumps({
    "date": "2026-06-10",
    "events": [
        {"start_time": "2026-06-10T09:00:00", "end_time": "2026-06-10T10:30:00",
         "place": {"id": "duomo", "name": "Duomo di Milano"}},
        {"time": "2026-06-10T12:00:00", "meal_slot": "lunch",
         "restaurant": {"id": "r1", "name": "Trattoria Milanese", "rating": 4.4}},
    ]
})


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL EXECUTION TESTS (mock A2A, test logic)
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolExecution:

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.agent.send_a2a_message", new_callable=AsyncMock)
    @patch("app.agent.extract_text_from_task")
    async def test_recommend_places_tool(self, mock_extract, mock_send):
        mock_send.return_value = {"artifacts": []}
        mock_extract.return_value = MOCK_PLACES_RESPONSE

        result = await _execute_tool("recommend_places", {
            "city": "Milan",
            "preferences": "museums, architecture",
        })

        mock_send.assert_called_once()
        assert "8000" in mock_send.call_args[1]["base_url"]  # Agent 1
        assert "Duomo" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.agent.send_a2a_message", new_callable=AsyncMock)
    @patch("app.agent.extract_text_from_task")
    async def test_cluster_places_tool(self, mock_extract, mock_send):
        mock_send.return_value = {"artifacts": []}
        mock_extract.return_value = MOCK_CLUSTERS_RESPONSE

        result = await _execute_tool("cluster_places", {
            "place_candidates_json": MOCK_PLACES_RESPONSE,
            "trip_start": "2026-06-10T09:00:00",
            "trip_end": "2026-06-11T22:00:00",
        })

        mock_send.assert_called_once()
        assert "clustered" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.agent.send_a2a_message", new_callable=AsyncMock)
    @patch("app.agent.extract_text_from_task")
    async def test_schedule_day_tool(self, mock_extract, mock_send):
        mock_send.return_value = {"artifacts": []}
        mock_extract.return_value = MOCK_SCHEDULE_RESPONSE

        result = await _execute_tool("schedule_day", {
            "cluster_json": json.dumps([{"id": "duomo", "name": "Duomo"}]),
            "day_date": "2026-06-10",
            "food_budget_per_day": 60.0,
        })

        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args[1]
        assert "8003" in call_kwargs["base_url"]  # Agent 3
        assert "2026-06-10" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_unknown_tool_returns_error(self):
        result = await _execute_tool("nonexistent_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.agent.send_a2a_message", new_callable=AsyncMock)
    async def test_tool_handles_agent_failure(self, mock_send):
        mock_send.side_effect = Exception("Connection refused")

        result = await _execute_tool("recommend_places", {"city": "Milan"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Connection refused" in parsed["error"]


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL AGENT LOOP TEST (mock LLM + mock tools)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_tool_call_response(tool_name: str, arguments: dict, call_id: str = "tc_1"):
    """Build a mock OpenAI ChatCompletion with a tool call."""
    mock_msg = MagicMock()
    mock_msg.content = None

    mock_tc = MagicMock()
    mock_tc.id = call_id
    mock_tc.function.name = tool_name
    mock_tc.function.arguments = json.dumps(arguments)
    mock_msg.tool_calls = [mock_tc]

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _make_text_response(text: str):
    """Build a mock OpenAI ChatCompletion with a final text response."""
    mock_msg = MagicMock()
    mock_msg.content = text
    mock_msg.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestAgentLoop:

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.agent._execute_tool", new_callable=AsyncMock)
    @patch("app.agent._get_client")
    async def test_simple_chat_no_tools(self, mock_client_fn, mock_tool):
        """User says 'hi' — LLM responds without calling any tools."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_text_response("Hello! Where would you like to go?")
        )
        mock_client_fn.return_value = mock_client

        response, history = await run_agent("Hi!")

        assert "Hello" in response
        mock_tool.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.agent._execute_tool", new_callable=AsyncMock)
    @patch("app.agent._get_client")
    async def test_tool_call_then_response(self, mock_client_fn, mock_tool):
        """LLM calls recommend_places, gets result, then responds with text."""
        mock_client = AsyncMock()

        # Round 1: LLM calls recommend_places
        # Round 2: LLM returns final text
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_tool_call_response("recommend_places", {"city": "Milan"}),
                _make_text_response("I found 2 great places in Milan: Duomo and Castello!"),
            ]
        )
        mock_client_fn.return_value = mock_client
        mock_tool.return_value = MOCK_PLACES_RESPONSE

        response, history = await run_agent("Find places to visit in Milan")

        assert "Milan" in response
        mock_tool.assert_called_once_with("recommend_places", {"city": "Milan"})

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.agent._execute_tool", new_callable=AsyncMock)
    @patch("app.agent._get_client")
    async def test_multi_tool_pipeline(self, mock_client_fn, mock_tool):
        """LLM calls recommend → cluster → schedule → responds."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_tool_call_response("recommend_places", {"city": "Milan"}, "tc_1"),
                _make_tool_call_response("cluster_places", {
                    "place_candidates_json": MOCK_PLACES_RESPONSE,
                    "trip_start": "2026-06-10", "trip_end": "2026-06-11",
                }, "tc_2"),
                _make_tool_call_response("schedule_day", {
                    "cluster_json": "[]", "day_date": "2026-06-10",
                }, "tc_3"),
                _make_text_response("Here's your Milan itinerary!"),
            ]
        )
        mock_client_fn.return_value = mock_client
        mock_tool.return_value = "{}"

        response, history = await run_agent("Plan a 2-day trip to Milan")

        assert "itinerary" in response.lower()
        assert mock_tool.call_count == 3

        # Verify the tools were called in order
        tool_names = [call.args[0] for call in mock_tool.call_args_list]
        assert tool_names == ["recommend_places", "cluster_places", "schedule_day"]


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER TEST (run_task through InMemoryStorage)
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorker:

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.run_agent", new_callable=AsyncMock)
    async def test_worker_completes_task(self, mock_agent):
        mock_agent.return_value = (
            "Here's your trip plan!",
            [{"role": "user", "content": "plan a trip"}],
        )

        storage = InMemoryStorage()
        broker  = InMemoryBroker()
        worker  = OrchestratorWorker(storage=storage, broker=broker)

        message = Message(
            role="user",
            kind="message",
            message_id=str(uuid.uuid4()),
            parts=[TextPart(text="Plan me a trip to Milan", kind="text")],
        )

        task = await storage.submit_task(str(uuid.uuid4()), message)
        await worker.run_task({"id": task["id"]})

        result = await storage.load_task(task["id"])
        assert result is not None
        assert result["status"]["state"] == "completed"
        assert len(result.get("artifacts", [])) >= 1

        mock_agent.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.run_agent", new_callable=AsyncMock)
    async def test_worker_handles_agent_failure(self, mock_agent):
        mock_agent.side_effect = Exception("LLM exploded")

        storage = InMemoryStorage()
        broker  = InMemoryBroker()
        worker  = OrchestratorWorker(storage=storage, broker=broker)

        message = Message(
            role="user",
            kind="message",
            message_id=str(uuid.uuid4()),
            parts=[TextPart(text="test", kind="text")],
        )

        task = await storage.submit_task(str(uuid.uuid4()), message)
        await worker.run_task({"id": task["id"]})

        result = await storage.load_task(task["id"])
        assert result["status"]["state"] == "failed"
