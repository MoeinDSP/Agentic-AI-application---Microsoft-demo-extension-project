"""
Mock Orchestrator — Clustering Agent CLI tester.

Simulates Agent 0 calling Agent 2 via both interfaces:
  - A2A JSON-RPC  (used by the orchestrator LLM tool-calling loop)
  - REST POST /cluster  (used by Agent 3 — Daily Scheduler)

Usage:
    python -m tests.mock_orchestrator                    # default (Milan, 2 days)
    python -m tests.mock_orchestrator --interface rest
    python -m tests.mock_orchestrator --interface a2a
    python -m tests.mock_orchestrator --days 3
"""
from __future__ import annotations

import argparse
import json
import time
import uuid

import requests

from app.core.config import settings
from app.models import ClusteringRequest, Location, PlaceCandidate

# ── Sample data ───────────────────────────────────────────────────────────────

MILAN_PLACES = [
    PlaceCandidate(id="duomo",        name="Duomo di Milano",      location=Location(latitude=45.4641, longitude=9.1919), rating=4.8),
    PlaceCandidate(id="castello",     name="Castello Sforzesco",   location=Location(latitude=45.4705, longitude=9.1794), rating=4.5),
    PlaceCandidate(id="navigli",      name="Navigli",              location=Location(latitude=45.4505, longitude=9.1751), rating=4.3),
    PlaceCandidate(id="brera",        name="Brera District",       location=Location(latitude=45.4721, longitude=9.1872), rating=4.4),
    PlaceCandidate(id="centrale",     name="Stazione Centrale",    location=Location(latitude=45.4862, longitude=9.2045), rating=3.9),
    PlaceCandidate(id="isola",        name="Isola District",       location=Location(latitude=45.4880, longitude=9.1860), rating=4.1),
    PlaceCandidate(id="porta_romana", name="Porta Romana",         location=Location(latitude=45.4527, longitude=9.2010), rating=4.0),
    PlaceCandidate(id="city_life",    name="CityLife",             location=Location(latitude=45.4771, longitude=9.1419), rating=4.2),
]


# ── Agent card ────────────────────────────────────────────────────────────────

def get_agent_card() -> dict:
    resp = requests.get(f"{settings.agent_url}/.well-known/agent.json", timeout=5)
    resp.raise_for_status()
    return resp.json()


# ── REST interface ────────────────────────────────────────────────────────────

def call_via_rest(req: ClusteringRequest) -> dict:
    resp = requests.post(
        f"{settings.agent_url}/cluster",
        data=req.model_dump_json(),
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── A2A interface ─────────────────────────────────────────────────────────────

def send_a2a_task(payload: str) -> str:
    body = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": payload}],
            },
        },
    }
    resp = requests.post(f"{settings.agent_url}/", json=body, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("result", {}).get("id")
    if not task_id:
        raise RuntimeError(f"No task id in response: {data}")
    return task_id


def poll_a2a_task(task_id: str, interval: float = 1.0, max_wait: float = 60.0) -> dict:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        body = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        resp = requests.post(f"{settings.agent_url}/", json=body, timeout=10)
        resp.raise_for_status()
        data  = resp.json()
        task  = data.get("result", data)
        state = task.get("status", {}).get("state", "unknown")

        print(f"  ↳ state: {state}")

        if state == "completed":
            return task
        if state in ("canceled", "failed"):
            raise RuntimeError(f"Task ended with state: {state}")

        time.sleep(interval)

    raise TimeoutError(f"Task {task_id} did not complete within {max_wait}s")


def extract_clusters_from_a2a(task: dict) -> list:
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "data":
                return part.get("data", {}).get("clustered_place_candidates", [])

    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                try:
                    clean = part.get("text", "").strip()
                    if clean.startswith("```"):
                        clean = "\n".join(clean.split("\n")[1:-1])
                    return json.loads(clean).get("clustered_place_candidates", [])
                except (json.JSONDecodeError, AttributeError):
                    pass
    return []


# ── Result printer ────────────────────────────────────────────────────────────

def print_clusters(clusters: list, num_days: int) -> None:
    if not clusters:
        print("\n⚠️  No clusters returned.\n")
        return

    print(f"\n{'─' * 60}")
    print(f"  📍 Clustered Places ({len(clusters)} day(s) planned of {num_days})")
    print(f"{'─' * 60}")

    for day_idx, cluster in enumerate(clusters, 1):
        print(f"\n  Day {day_idx} — {len(cluster)} place(s):")
        for place in cluster:
            name   = place.get("name", "Unknown")
            rating = place.get("rating") or "N/A"
            lat    = place.get("location", {}).get("latitude", 0)
            lon    = place.get("location", {}).get("longitude", 0)
            print(f"    • {name}  (rating: {rating}, lat={lat:.4f}, lon={lon:.4f})")

    print(f"\n{'─' * 60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Orchestrator — test Agent 2 via A2A or REST")
    parser.add_argument("--interface", choices=["rest", "a2a"], default="rest")
    parser.add_argument("--days", type=int, default=2, help="Trip duration in days")
    args = parser.parse_args()

    from datetime import datetime, timedelta
    trip_start = datetime(2026, 6, 10, 9, 0)
    trip_end   = trip_start + timedelta(days=args.days)

    req = ClusteringRequest(
        trip_start=trip_start,
        trip_end=trip_end,
        place_candidates=MILAN_PLACES,
    )

    print(f"\n🚀 Mock Orchestrator → Agent 2 at {settings.agent_url}")
    print(f"   Interface : {args.interface.upper()}")
    print(f"   Trip      : {trip_start.date()} → {trip_end.date()} ({args.days} day(s))")
    print(f"   Places    : {len(MILAN_PLACES)}\n")

    card = get_agent_card()
    print(f"✅ Agent card: {card.get('name')} v{card.get('version')}\n")

    clusters: list = []

    if args.interface == "rest":
        print("📤 Calling REST /cluster ...")
        result = call_via_rest(req)
        clusters = result.get("clustered_place_candidates", [])
        print(f"✅ REST response received\n")

    else:
        print("📤 Sending A2A task ...")
        task_id = send_a2a_task(req.model_dump_json())
        print(f"   Task ID: {task_id}\n")
        print("⏳ Polling for result ...")
        task    = poll_a2a_task(task_id)
        clusters = extract_clusters_from_a2a(task)

    print_clusters(clusters, args.days)


if __name__ == "__main__":
    main()
