"""Minimal A2A client to smoke-test the single-day-plan-scheduler agent.

Usage:
    single-day-plan-scheduler/.venv/bin/python single-day-plan-scheduler/test_a2a_client.py
    # or against another host:
    BASE_URL=http://localhost:8003 .../python test_a2a_client.py
"""

from __future__ import annotations

import json
import os
import time
import uuid

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8003").rstrip("/")

# A DaySchedulingRequest payload (sent as the message text).
REQUEST = {
    "places": [
        {
            "id": "duomo",
            "name": "Duomo di Milano",
            "location": {"latitude": 45.4641, "longitude": 9.1919},
            "estimated_visit_duration_minutes": 90,
            "priority_score": 0.9,
        },
        {
            "id": "castello",
            "name": "Castello Sforzesco",
            "location": {"latitude": 45.4705, "longitude": 9.1796},
            "estimated_visit_duration_minutes": 75,
            "priority_score": 0.6,
        },
    ],
    "day_start": "2026-06-01T09:00:00",
    "day_end": "2026-06-01T21:00:00",
    "food_budget_per_day": 60,
    "preferences": ["italian"],
    "acceptable_transport_modes": ["walking"],
}


def main() -> None:
    message_id = f"test-{uuid.uuid4().hex}"
    send = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": json.dumps(REQUEST)}],
                "kind": "message",
                "messageId": message_id,
            },
            "configuration": {"acceptedOutputModes": ["application/json"]},
        },
        "id": message_id,
    }

    with httpx.Client(timeout=300) as client:
        r = client.post(f"{BASE_URL}/", json=send)
        r.raise_for_status()
        result = r.json().get("result", {})
        task_id = result.get("id")
        print(f"submitted task: {task_id} (state={result.get('status', {}).get('state')})")

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            r = client.post(
                f"{BASE_URL}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {"id": task_id},
                    "id": f"{task_id}-poll",
                },
            )
            r.raise_for_status()
            task = r.json().get("result", {})
            state = task.get("status", {}).get("state")
            print(f"  state={state}")
            if state in {"completed", "failed", "canceled", "rejected"}:
                print("\n=== artifacts ===")
                print(json.dumps(task.get("artifacts", []), indent=2)[:4000])
                if state != "completed":
                    print("\n=== status message ===")
                    print(json.dumps(task.get("status", {}), indent=2)[:2000])
                return
            time.sleep(2)
        print("timed out waiting for task")


if __name__ == "__main__":
    main()
