"""
Mock Agent 3 — Daily Scheduler (CLI)
Synchronous version — uses `requests` to avoid httpx/anyio issues on Python 3.14+

Usage:
    python -m tests.mock_agent3                        # default (lunch, Milan Duomo)
    python -m tests.mock_agent3 --scenario breakfast
    python -m tests.mock_agent3 --scenario dinner --budget 60
    python -m tests.mock_agent3 --lat 48.8566 --lon 2.3522 --scenario lunch
"""
from __future__ import annotations

import argparse
import json
import time
import uuid

import requests

from app.core.config import settings
from app.models import FoodRecommenderInput, Location, MealSlot

# ── Predefined test scenarios ─────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "breakfast": {
        "timeofday": MealSlot.breakfast,
        "searchcenter": Location(latitude=45.4654, longitude=9.1859),
        "searchradiusmeters": 600,
        "budgetpermealperperson": 12.0,
        "preferences": ["cafe", "bakery"],
    },
    "lunch": {
        "timeofday": MealSlot.lunch,
        "searchcenter": Location(latitude=45.4654, longitude=9.1859),
        "searchradiusmeters": 800,
        "budgetpermealperperson": 25.0,
        "preferences": ["italian"],
    },
    "dinner": {
        "timeofday": MealSlot.dinner,
        "searchcenter": Location(latitude=45.4654, longitude=9.1859),
        "searchradiusmeters": 1000,
        "budgetpermealperperson": 45.0,
        "preferences": ["italian", "wine"],
    },
}


# ── A2A helpers ───────────────────────────────────────────────────────────────

def get_agent_card() -> dict:
    resp = requests.get(
        f"{settings.agent_url}/.well-known/agent.json",
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


def send_task(payload: str) -> str:
    """POST message/send, return the task id assigned by the server."""
    body = {
        "jsonrpc": "2.0",
        "id":      str(uuid.uuid4()),
        "method":  "message/send",
        "params": {
            "message": {
                "role":      "user",
                "kind":      "message",
                "messageId": str(uuid.uuid4()),
                "parts":     [{"kind": "text", "text": payload}],
            },
        },
    }
    resp = requests.post(
        f"{settings.agent_url}/",
        json=body,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    task_id = data.get("result", {}).get("id")
    if not task_id:
        raise RuntimeError(f"No task id in response: {data}")
    return task_id


def poll_task(
    task_id: str,
    interval: float = 1.5,
    max_wait: float = 120.0,
) -> dict:
    """Poll tasks/get until state is completed, canceled, or failed."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        body = {
            "jsonrpc": "2.0",
            "id":      str(uuid.uuid4()),
            "method":  "tasks/get",
            "params":  {"id": task_id},
        }
        resp = requests.post(
            f"{settings.agent_url}/",
            json=body,
            timeout=10,
        )
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


# ── Result printer ────────────────────────────────────────────────────────────

def print_results(task: dict) -> None:
    candidates = None

    # Try DataPart artifact first
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "data":
                candidates = part.get("data", {}).get("restaurantcandidates")
                break
        if candidates is not None:
            break

    # Fall back to TextPart
    if candidates is None:
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    try:
                        clean = part.get("text", "").strip()
                        if clean.startswith("```"):
                            clean = "\n".join(clean.split("\n")[1:-1])
                        candidates = json.loads(clean).get("restaurantcandidates", [])
                    except (json.JSONDecodeError, AttributeError):
                        candidates = []

    if not candidates:
        print("\n⚠️  No restaurant candidates returned.\n")
        return

    print(f"\n{'─' * 60}")
    print(f"  🍽️  Restaurant Candidates ({len(candidates)} found)")
    print(f"{'─' * 60}")

    for i, r in enumerate(candidates, 1):
        rating   = r.get("rating") or "N/A"
        price    = "€" * int(r.get("price_level") or 0) or "N/A"
        cuisines = ", ".join(r.get("cuisines") or []) or "N/A"
        address  = r.get("location", {}).get("address") or "N/A"
        summary  = r.get("summary") or ""

        print(f"\n  {i}. {r['name']}")
        print(f"     Rating   : {rating} ⭐")
        print(f"     Price    : {price}")
        print(f"     Cuisines : {cuisines}")
        print(f"     Address  : {address}")
        if summary:
            print(f"     Summary  : {summary}")

    print(f"\n{'─' * 60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Agent 3 — test Agent 4 via A2A")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="lunch",
    )
    parser.add_argument("--lat",    type=float)
    parser.add_argument("--lon",    type=float)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario].copy()

    if args.lat is not None or args.lon is not None:
        lat = args.lat or scenario["searchcenter"].latitude
        lon = args.lon or scenario["searchcenter"].longitude
        scenario["searchcenter"] = Location(latitude=lat, longitude=lon)

    if args.radius is not None:
        scenario["searchradiusmeters"] = args.radius
    if args.budget is not None:
        scenario["budgetpermealperperson"] = args.budget

    inp     = FoodRecommenderInput(**scenario)
    payload = inp.model_dump_json()

    print(f"\n🚀 Mock Agent 3 → sending task to Agent 4 at {settings.agent_url}")
    print(f"   Scenario : {args.scenario}")
    print(f"   Payload  : {payload}\n")

    # 1. Discover agent card
    card = get_agent_card()
    print(f"✅ Agent card: {card.get('name')} v{card.get('version')}\n")

    # 2. Send task
    print("📤 Sending task...")
    task_id = send_task(payload)
    print(f"   Task ID: {task_id}\n")

    # 3. Poll until done
    print("⏳ Polling for result...")
    task = poll_task(task_id)

    # 4. Print results
    print_results(task)


if __name__ == "__main__":
    main()