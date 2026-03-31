from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv

from app.core.config import settings
from app.models import FoodRecommenderInput, Location, MealSlot

load_dotenv()

# ── Predefined test scenarios ─────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "breakfast": {
        "timeofday": MealSlot.breakfast,
        "searchcenter": Location(latitude=45.4654, longitude=9.1859),  # Milan Duomo
        "searchradiusmeters": 600,
        "budgetpermealperperson": 12.0,
        "preferences": ["cafe", "bakery"],
    },
    "lunch": {
        "timeofday": MealSlot.lunch,
        "searchcenter": Location(latitude=45.4654, longitude=9.1859),  # Milan Duomo
        "searchradiusmeters": 800,
        "budgetpermealperperson": 25.0,
        "preferences": ["italian"],
    },
    "dinner": {
        "timeofday": MealSlot.dinner,
        "searchcenter": Location(latitude=45.4654, longitude=9.1859),  # Milan Duomo
        "searchradiusmeters": 1000,
        "budgetpermealperperson": 45.0,
        "preferences": ["italian", "wine"],
    },
}


# ── A2A client helpers ────────────────────────────────────────────────────────

async def send_task(client: httpx.AsyncClient, payload: str) -> str:
    """POST /tasks/send and return the task id."""
    task_id = str(uuid.uuid4())
    body = {
        "id": task_id,
        "message": {
            "role": "user",
            "kind": "message",
            "message_id": str(uuid.uuid4()),
            "parts": [{"kind": "text", "text": payload}],
        },
    }
    resp = await client.post(
        f"{settings.agent_url}/tasks/send",
        json=body,
        timeout=15,
    )
    resp.raise_for_status()
    return task_id


async def poll_task(
    client: httpx.AsyncClient,
    task_id: str,
    interval: float = 1.0,
    max_wait: float = 60.0,
) -> dict[str, Any]:
    """Poll GET /tasks/{id} until state is completed, canceled, or failed."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        resp = await client.get(
            f"{settings.agent_url}/tasks/{task_id}",
            timeout=10,
        )
        resp.raise_for_status()
        task = resp.json()
        state = task.get("status", {}).get("state", "")

        print(f"  ↳ state: {state}")

        if state == "completed":
            return task
        if state in ("canceled", "failed"):
            raise RuntimeError(f"Task ended with state: {state}")

        await asyncio.sleep(interval)

    raise TimeoutError(f"Task {task_id} did not complete within {max_wait}s")


# ── Result printer ────────────────────────────────────────────────────────────

def print_results(task: dict[str, Any]) -> None:
    """Pretty-print restaurant candidates from the completed task."""
    # Try DataPart artifact first (structured), fall back to TextPart
    candidates = None

    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "data":
                data = part.get("data", {})
                candidates = data.get("restaurantcandidates")
                break
        if candidates is not None:
            break

    if candidates is None:
        # Fall back to parsing text artifact
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    try:
                        text = part.get("text", "")
                        clean = text.strip()
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
        rating     = r.get("rating") or "N/A"
        price      = "€" * int(r.get("price_level") or 0) or "N/A"
        cuisines   = ", ".join(r.get("cuisines") or []) or "N/A"
        address    = r.get("location", {}).get("address") or "N/A"
        summary    = r.get("summary") or ""

        print(f"\n  {i}. {r['name']}")
        print(f"     Rating   : {rating} ⭐")
        print(f"     Price    : {price}")
        print(f"     Cuisines : {cuisines}")
        print(f"     Address  : {address}")
        if summary:
            print(f"     Summary  : {summary}")

    print(f"\n{'─' * 60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Agent 3 — test Agent 4 via A2A")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="lunch",
        help="Predefined meal scenario (default: lunch)",
    )
    parser.add_argument("--lat",    type=float, help="Override latitude")
    parser.add_argument("--lon",    type=float, help="Override longitude")
    parser.add_argument("--radius", type=int,   help="Override search radius in metres")
    parser.add_argument("--budget", type=float, help="Override budget per person in EUR")
    args = parser.parse_args()

    # Build input from scenario + CLI overrides
    scenario = SCENARIOS[args.scenario].copy()

    if args.lat is not None or args.lon is not None:
        lat = args.lat or scenario["searchcenter"].latitude
        lon = args.lon or scenario["searchcenter"].longitude
        scenario["searchcenter"] = Location(latitude=lat, longitude=lon)

    if args.radius is not None:
        scenario["searchradiusmeters"] = args.radius

    if args.budget is not None:
        scenario["budgetpermealperperson"] = args.budget

    inp = FoodRecommenderInput(**scenario)
    payload = inp.model_dump_json()

    print(f"\n🚀 Mock Agent 3 → sending task to Agent 4 at {settings.agent_url}")
    print(f"   Scenario : {args.scenario}")
    print(f"   Payload  : {payload}\n")

    async with httpx.AsyncClient() as client:
        # 1. Discover agent card
        card_resp = await client.get(
            f"{settings.agent_url}/.well-known/agent.json",
            timeout=5,
        )
        card_resp.raise_for_status()
        card = card_resp.json()
        print(f"✅ Agent card: {card.get('name')} v{card.get('version')}\n")

        # 2. Send task
        print("📤 Sending task...")
        task_id = await send_task(client, payload)
        print(f"   Task ID: {task_id}\n")

        # 3. Poll until done
        print("⏳ Polling for result...")
        task = await poll_task(client, task_id)

    # 4. Print results
    print_results(task)


if __name__ == "__main__":
    asyncio.run(main())