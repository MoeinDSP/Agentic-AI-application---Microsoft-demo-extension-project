"""
Test Client — Orchestrator (CLI)
Synchronous version — uses `requests` to avoid httpx/anyio issues.

Usage:
    python -m tests.test_client                          # default Milan 3-day trip
    python -m tests.test_client --city Paris --days 5
    python -m tests.test_client --city Rome --budget 800 --reason honeymoon
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timedelta

import requests

from app.core.config import settings
from app.models import BudgetConstraint, Location, TripRequest


# ── Predefined test scenarios ─────────────────────────────────────────────────

DEFAULT_SCENARIOS: dict[str, dict] = {
    "milan_3day": {
        "city": "Milan",
        "trip_start": datetime(2026, 6, 10, 9, 0),
        "trip_end": datetime(2026, 6, 12, 22, 0),
        "location": Location(latitude=45.4654, longitude=9.1859),
        "budget": BudgetConstraint(total_budget=600.0),
        "trip_reason": "city break",
        "preferences": ["museums", "architecture", "italian food"],
    },
    "paris_weekend": {
        "city": "Paris",
        "trip_start": datetime(2026, 7, 4, 9, 0),
        "trip_end": datetime(2026, 7, 6, 22, 0),
        "location": Location(latitude=48.8566, longitude=2.3522),
        "budget": BudgetConstraint(total_budget=1000.0),
        "trip_reason": "romantic getaway",
        "preferences": ["art", "fine dining", "river walks"],
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
    """POST message/send, return the task id."""
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
    interval: float = 2.0,
    max_wait: float = 300.0,
) -> dict:
    """Poll tasks/get until terminal state."""
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
    # Try DataPart first
    itinerary = None
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "data":
                itinerary = part.get("data")
                break
        if itinerary:
            break

    # Fall back to TextPart
    if itinerary is None:
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    try:
                        clean = part.get("text", "").strip()
                        if clean.startswith("```"):
                            clean = "\n".join(clean.split("\n")[1:-1])
                        itinerary = json.loads(clean)
                    except (json.JSONDecodeError, AttributeError):
                        pass

    if not itinerary:
        print("\n⚠️  No itinerary returned.\n")
        return

    schedules = itinerary.get("daily_schedules", [])
    warnings  = itinerary.get("warnings", [])
    derived   = itinerary.get("derived", {})

    print(f"\n{'═' * 60}")
    print(f"  🗓️  Trip Itinerary — {len(schedules)} day(s)")
    print(f"  Budget split: {derived.get('activity_budget', 'N/A')} EUR activity "
          f"/ {derived.get('food_budget_total', 'N/A')} EUR food")
    print(f"{'═' * 60}")

    for day in schedules:
        print(f"\n  📅 {day.get('date', '?')}")
        print(f"  {'─' * 50}")

        for i, event in enumerate(day.get("events", []), 1):
            if "place" in event:
                place = event["place"]
                start = event.get("start_time", "?")
                end   = event.get("end_time", "?")
                print(f"  {i}. 🏛️  {place.get('name', '?')}")
                print(f"       {start} → {end}")
                if place.get("category"):
                    print(f"       Category: {place['category']}")
            elif "restaurant" in event:
                rest = event["restaurant"]
                t    = event.get("time", "?")
                slot = event.get("meal_slot", "?")
                print(f"  {i}. 🍽️  {rest.get('name', '?')} ({slot})")
                print(f"       {t}")
                rating = rest.get("rating", "N/A")
                price  = "€" * int(rest.get("price_level") or 0) or "N/A"
                print(f"       Rating: {rating} ⭐  Price: {price}")

    if warnings:
        print(f"\n  ⚠️  Warnings:")
        for w in warnings:
            print(f"     - {w}")

    print(f"\n{'═' * 60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test Client — Orchestrator Agent")
    parser.add_argument("--scenario", choices=list(DEFAULT_SCENARIOS.keys()), default="milan_3day")
    parser.add_argument("--city",    type=str)
    parser.add_argument("--days",    type=int)
    parser.add_argument("--budget",  type=float)
    parser.add_argument("--reason",  type=str)
    args = parser.parse_args()

    scenario = DEFAULT_SCENARIOS[args.scenario].copy()

    if args.city:
        scenario["city"] = args.city
    if args.days:
        scenario["trip_end"] = scenario["trip_start"] + timedelta(days=args.days)
    if args.budget:
        scenario["budget"] = BudgetConstraint(total_budget=args.budget)
    if args.reason:
        scenario["trip_reason"] = args.reason

    request = TripRequest(**scenario)
    payload = request.model_dump_json()

    print(f"\n🚀 Sending trip request to Orchestrator at {settings.agent_url}")
    print(f"   City    : {request.city}")
    print(f"   Dates   : {request.trip_start} → {request.trip_end}")
    print(f"   Budget  : {request.budget}")
    print(f"   Reason  : {request.trip_reason}")
    print(f"   Prefs   : {request.preferences}\n")

    # 1. Agent card
    card = get_agent_card()
    print(f"✅ Agent card: {card.get('name')} v{card.get('version')}\n")

    # 2. Send
    print("📤 Sending task...")
    task_id = send_task(payload)
    print(f"   Task ID: {task_id}\n")

    # 3. Poll
    print("⏳ Polling for result (this may take a few minutes)...")
    task = poll_task(task_id, max_wait=300)

    # 4. Print
    print_results(task)


if __name__ == "__main__":
    main()
