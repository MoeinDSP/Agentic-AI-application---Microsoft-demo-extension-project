from __future__ import annotations

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import settings
from app.models import ClusteringRequest, ClusteringResponse, Location, PlaceCandidate
from app.services.clustering import cluster_places, _sq_dist


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def milan_places() -> list[PlaceCandidate]:
    """8 Milan landmarks spread across the city — 4 north, 4 south."""
    return [
        PlaceCandidate(id="duomo",     name="Duomo di Milano",      location=Location(latitude=45.4641, longitude=9.1919), rating=4.8),
        PlaceCandidate(id="castello",  name="Castello Sforzesco",    location=Location(latitude=45.4705, longitude=9.1794), rating=4.5),
        PlaceCandidate(id="navigli",   name="Navigli",               location=Location(latitude=45.4505, longitude=9.1751), rating=4.3),
        PlaceCandidate(id="brera",     name="Brera District",        location=Location(latitude=45.4721, longitude=9.1872), rating=4.4),
        PlaceCandidate(id="centrale",  name="Stazione Centrale",     location=Location(latitude=45.4862, longitude=9.2045), rating=3.9),
        PlaceCandidate(id="isola",     name="Isola District",        location=Location(latitude=45.4880, longitude=9.1860), rating=4.1),
        PlaceCandidate(id="porta_romana", name="Porta Romana",       location=Location(latitude=45.4527, longitude=9.2010), rating=4.0),
        PlaceCandidate(id="city_life", name="CityLife",              location=Location(latitude=45.4771, longitude=9.1419), rating=4.2),
    ]


@pytest.fixture
def two_day_request(milan_places) -> ClusteringRequest:
    return ClusteringRequest(
        trip_start=datetime(2026, 6, 10, 0, 0),
        trip_end=datetime(2026, 6, 12, 0, 0),   # exactly 48 h → 2 days
        place_candidates=milan_places,
    )


@pytest.fixture
def one_day_request(milan_places) -> ClusteringRequest:
    return ClusteringRequest(
        trip_start=datetime(2026, 6, 10, 9, 0),
        trip_end=datetime(2026, 6, 10, 22, 0),
        place_candidates=milan_places[:4],
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def agent2_is_running() -> bool:
    try:
        resp = httpx.get(f"{settings.agent_url}/.well-known/agent.json", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ── Unit: models ──────────────────────────────────────────────────────────────

class TestModels:

    @pytest.mark.unit
    def test_place_candidate_defaults(self):
        place = PlaceCandidate(
            id="x",
            name="Test",
            location=Location(latitude=45.0, longitude=9.0),
        )
        assert place.estimated_visit_duration_minutes == 60
        assert place.priority_score == 0.0
        assert place.opening_hours == []

    @pytest.mark.unit
    def test_clustering_request_valid(self, two_day_request):
        assert len(two_day_request.place_candidates) == 8
        assert two_day_request.trip_start < two_day_request.trip_end

    @pytest.mark.unit
    def test_clustering_request_json_roundtrip(self, two_day_request):
        serialised   = two_day_request.model_dump_json()
        deserialised = ClusteringRequest.model_validate_json(serialised)
        assert len(deserialised.place_candidates) == 8
        assert deserialised.trip_start == two_day_request.trip_start

    @pytest.mark.unit
    def test_clustering_response_structure(self, milan_places):
        resp = ClusteringResponse(
            clustered_place_candidates=[[milan_places[0], milan_places[1]]]
        )
        assert len(resp.clustered_place_candidates) == 1
        assert len(resp.clustered_place_candidates[0]) == 2


# ── Unit: clustering algorithm ────────────────────────────────────────────────

class TestClusteringAlgorithm:

    @pytest.mark.unit
    def test_empty_input_returns_empty(self):
        result = cluster_places(
            [],
            datetime(2026, 6, 10),
            datetime(2026, 6, 12),
        )
        assert result == []

    @pytest.mark.unit
    def test_single_place_returns_one_cluster(self):
        place = PlaceCandidate(
            id="p1", name="Place 1",
            location=Location(latitude=45.0, longitude=9.0),
        )
        result = cluster_places(
            [place],
            datetime(2026, 6, 10),
            datetime(2026, 6, 12),
        )
        assert len(result) == 1
        assert result[0][0].id == "p1"

    @pytest.mark.unit
    def test_one_day_trip_returns_one_cluster(self, milan_places):
        result = cluster_places(
            milan_places,
            datetime(2026, 6, 10, 9, 0),
            datetime(2026, 6, 10, 22, 0),
        )
        assert len(result) == 1
        assert sum(len(c) for c in result) == len(milan_places)

    @pytest.mark.unit
    def test_all_places_are_preserved(self, milan_places):
        result = cluster_places(
            milan_places,
            datetime(2026, 6, 10),
            datetime(2026, 6, 12),
        )
        all_ids = {p.id for cluster in result for p in cluster}
        expected_ids = {p.id for p in milan_places}
        assert all_ids == expected_ids

    @pytest.mark.unit
    def test_two_day_trip_produces_two_clusters(self, milan_places):
        result = cluster_places(
            milan_places,
            datetime(2026, 6, 10),
            datetime(2026, 6, 12),
        )
        assert len(result) == 2

    @pytest.mark.unit
    def test_num_clusters_capped_at_num_places(self):
        places = [
            PlaceCandidate(id=f"p{i}", name=f"Place {i}",
                           location=Location(latitude=45.0 + i * 0.01, longitude=9.0))
            for i in range(3)
        ]
        # 10-day trip but only 3 places → 3 clusters max
        result = cluster_places(
            places,
            datetime(2026, 6, 1),
            datetime(2026, 6, 11),
        )
        assert len(result) <= 3

    @pytest.mark.unit
    def test_geographically_coherent_clusters(self):
        """Northern and southern places should fall into separate clusters."""
        north = [
            PlaceCandidate(id=f"n{i}", name=f"North {i}",
                           location=Location(latitude=45.50 + i * 0.001, longitude=9.19))
            for i in range(4)
        ]
        south = [
            PlaceCandidate(id=f"s{i}", name=f"South {i}",
                           location=Location(latitude=45.43 + i * 0.001, longitude=9.19))
            for i in range(4)
        ]
        result = cluster_places(
            north + south,
            datetime(2026, 6, 10),
            datetime(2026, 6, 12),
        )
        assert len(result) == 2
        cluster_ids = [frozenset(p.id for p in c) for c in result]
        north_ids = frozenset(p.id for p in north)
        south_ids = frozenset(p.id for p in south)
        assert north_ids in cluster_ids
        assert south_ids in cluster_ids

    @pytest.mark.unit
    def test_num_days_computation_rounds_up(self):
        # 1.5 days → 2 days → 2 clusters (if enough places)
        places = [
            PlaceCandidate(id=f"p{i}", name=f"P{i}",
                           location=Location(latitude=45.0 + i * 0.1, longitude=9.0))
            for i in range(6)
        ]
        result = cluster_places(
            places,
            datetime(2026, 6, 10, 9, 0),
            datetime(2026, 6, 11, 21, 0),   # 36h = 1.5 days → ceil = 2
        )
        assert len(result) == 2

    @pytest.mark.unit
    def test_deterministic_with_same_seed(self, milan_places):
        r1 = cluster_places(milan_places, datetime(2026, 6, 10), datetime(2026, 6, 13), random_seed=0)
        r2 = cluster_places(milan_places, datetime(2026, 6, 10), datetime(2026, 6, 13), random_seed=0)
        ids1 = [sorted(p.id for p in c) for c in sorted(r1, key=lambda c: c[0].id)]
        ids2 = [sorted(p.id for p in c) for c in sorted(r2, key=lambda c: c[0].id)]
        assert ids1 == ids2

    @pytest.mark.unit
    def test_sq_dist_helper(self):
        assert _sq_dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(25.0)
        assert _sq_dist((1.0, 1.0), (1.0, 1.0)) == pytest.approx(0.0)


# ── Unit: worker ──────────────────────────────────────────────────────────────

class TestWorker:

    @pytest.mark.unit
    def test_parse_result_valid_json(self):
        from app.worker import ClusteringWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = ClusteringWorker(storage=InMemoryStorage(), broker=InMemoryBroker())
        payload = json.dumps({"clustered_place_candidates": [[{"id": "p1", "name": "Place 1"}]]})
        result = worker._parse_result(payload)
        assert "clustered_place_candidates" in result
        assert len(result["clustered_place_candidates"]) == 1

    @pytest.mark.unit
    def test_parse_result_strips_code_fences(self):
        from app.worker import ClusteringWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = ClusteringWorker(storage=InMemoryStorage(), broker=InMemoryBroker())
        payload = json.dumps({"clustered_place_candidates": []})
        fenced = f"```json\n{payload}\n```"
        result = worker._parse_result(fenced)
        assert "clustered_place_candidates" in result

    @pytest.mark.unit
    def test_parse_result_invalid_json_raises(self):
        from app.worker import ClusteringWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        worker = ClusteringWorker(storage=InMemoryStorage(), broker=InMemoryBroker())
        with pytest.raises(Exception):
            worker._parse_result("this is not valid json at all")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_worker_completes_task(self, two_day_request):
        from app.worker import ClusteringWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage
        from fasta2a.schema import Message, TextPart

        storage = InMemoryStorage()
        broker  = InMemoryBroker()
        worker  = ClusteringWorker(storage=storage, broker=broker)

        message = Message(
            role="user",
            kind="message",
            message_id=str(uuid.uuid4()),
            parts=[TextPart(text=two_day_request.model_dump_json(), kind="text")],
        )

        task = await storage.submit_task(str(uuid.uuid4()), message)
        await worker.run_task({"id": task["id"]})

        result = await storage.load_task(task["id"])
        assert result is not None
        assert result["status"]["state"] == "completed"
        assert len(result.get("artifacts", [])) >= 2

        # Verify DataPart artifact contains clustered candidates
        data_artifact = next(
            (a for a in result["artifacts"] if a.get("name") == "clustered_places"),
            None,
        )
        assert data_artifact is not None
        data_parts = [p for p in data_artifact["parts"] if p.get("kind") == "data"]
        assert len(data_parts) == 1
        assert "clustered_place_candidates" in data_parts[0]["data"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch("app.worker.run_agent", new_callable=AsyncMock)
    async def test_worker_fails_when_agent_raises(self, mock_run_agent):
        from app.worker import ClusteringWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage
        from fasta2a.schema import Message, TextPart

        mock_run_agent.side_effect = Exception("LLM exploded")

        storage = InMemoryStorage()
        broker  = InMemoryBroker()
        worker  = ClusteringWorker(storage=storage, broker=broker)

        message = Message(
            role="user",
            kind="message",
            message_id=str(uuid.uuid4()),
            parts=[TextPart(text="any input", kind="text")],
        )

        task = await storage.submit_task(str(uuid.uuid4()), message)
        await worker.run_task({"id": task["id"]})

        result = await storage.load_task(task["id"])
        assert result["status"]["state"] == "failed"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_worker_cancel(self, two_day_request):
        from app.worker import ClusteringWorker
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage
        from fasta2a.schema import Message, TextPart

        storage = InMemoryStorage()
        broker  = InMemoryBroker()
        worker  = ClusteringWorker(storage=storage, broker=broker)

        message = Message(
            role="user",
            kind="message",
            message_id=str(uuid.uuid4()),
            parts=[TextPart(text=two_day_request.model_dump_json(), kind="text")],
        )

        task = await storage.submit_task(str(uuid.uuid4()), message)
        await worker.cancel_task({"id": task["id"]})

        result = await storage.load_task(task["id"])
        assert result["status"]["state"] == "canceled"


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(
    not agent2_is_running(),
    reason="Agent 2 is not running at AGENT_URL",
)
class TestAgent2Integration:

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
    async def test_rest_cluster_endpoint(self, two_day_request):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.agent_url}/cluster",
                content=two_day_request.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "clustered_place_candidates" in data
        clusters = data["clustered_place_candidates"]
        assert len(clusters) == 2
        all_ids = {p["id"] for c in clusters for p in c}
        assert all_ids == {p.id for p in two_day_request.place_candidates}

    @pytest.mark.asyncio
    async def test_rest_cluster_empty_places(self):
        payload = {
            "trip_start": "2026-06-10T09:00:00",
            "trip_end": "2026-06-12T22:00:00",
            "place_candidates": [],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.agent_url}/cluster",
                json=payload,
                timeout=10,
            )
        assert resp.status_code == 200
        assert resp.json()["clustered_place_candidates"] == []

    @pytest.mark.asyncio
    async def test_a2a_round_trip(self, two_day_request):
        import asyncio

        send_body = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "kind": "message",
                    "messageId": str(uuid.uuid4()),
                    "parts": [{"kind": "text", "text": two_day_request.model_dump_json()}],
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

            for _ in range(30):
                poll_body = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tasks/get",
                    "params": {"id": task_id},
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

                await asyncio.sleep(1)
            else:
                pytest.fail("Task did not complete within 30s")

        assert len(task.get("artifacts", [])) >= 2
