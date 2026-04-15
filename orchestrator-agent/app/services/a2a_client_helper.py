"""
Helpers for calling downstream A2A agents (Agent 1 and Agent 4)
and the clustering REST endpoint (Agent 2).

Uses JSON-RPC over HTTP — the same protocol as fasta2a client.py and
the mock_agent3.py pattern from the food-recommender agent.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import httpx

from app.core.config import settings


# ── A2A JSON-RPC helpers ──────────────────────────────────────────────────────

def _jsonrpc_body(method: str, params: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id":      str(uuid.uuid4()),
        "method":  method,
        "params":  params,
    }


async def call_a2a_agent(
    base_url: str,
    payload: str,
    poll_interval: float = 1.5,
    max_wait: float = 120.0,
) -> dict:
    """
    Send a text message to an A2A agent via JSON-RPC ``message/send``,
    poll ``tasks/get`` until the task reaches a terminal state, and
    return the full task dict (including artifacts).

    Raises:
        RuntimeError  — if the task fails, is canceled, or times out.
        httpx.HTTPStatusError — on transport-level errors.
    """
    send_body = _jsonrpc_body("message/send", {
        "message": {
            "role":      "user",
            "kind":      "message",
            "messageId": str(uuid.uuid4()),
            "parts":     [{"kind": "text", "text": payload}],
        },
    })

    async with httpx.AsyncClient(timeout=15) as client:
        # ── 1. Send ───────────────────────────────────────────────────────
        send_resp = await client.post(f"{base_url}/", json=send_body)
        send_resp.raise_for_status()
        send_data = send_resp.json()

        task_id = send_data.get("result", {}).get("id")
        if not task_id:
            raise RuntimeError(f"No task id in A2A response: {send_data}")

        # ── 2. Poll ───────────────────────────────────────────────────────
        deadline = asyncio.get_event_loop().time() + max_wait

        while asyncio.get_event_loop().time() < deadline:
            poll_body = _jsonrpc_body("tasks/get", {"id": task_id})
            poll_resp = await client.post(f"{base_url}/", json=poll_body, timeout=10)
            poll_resp.raise_for_status()

            task = poll_resp.json().get("result", poll_resp.json())
            state = task.get("status", {}).get("state", "unknown")

            if state == "completed":
                return task
            if state in ("canceled", "failed"):
                raise RuntimeError(
                    f"A2A task {task_id} at {base_url} ended with state: {state}"
                )

            await asyncio.sleep(poll_interval)

        raise RuntimeError(
            f"A2A task {task_id} at {base_url} did not complete within {max_wait}s"
        )


# ── Artifact extraction ──────────────────────────────────────────────────────

def extract_data_artifact(task: dict, artifact_name: str | None = None) -> dict | None:
    """
    Pull a structured DataPart from the task's artifacts.
    If ``artifact_name`` is given, match on it; otherwise return the
    first DataPart found.
    """
    for artifact in task.get("artifacts", []):
        if artifact_name and artifact.get("name") != artifact_name:
            continue
        for part in artifact.get("parts", []):
            if part.get("kind") == "data":
                return part.get("data")
    return None


def extract_text_artifact(task: dict) -> str:
    """Return the first TextPart content from any artifact."""
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                return part.get("text", "")
    return ""


def parse_json_from_text(text: str) -> dict | None:
    """Attempt to parse JSON, stripping markdown fences if present."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:-1])
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return None


# ── Agent-specific callers ────────────────────────────────────────────────────

async def call_place_recommender(
    city: str,
    trip_start: str,
    trip_end: str,
    budget: float | None = None,
    trip_reason: str | None = None,
    preferences: list[str] | None = None,
) -> list[dict]:
    """
    Call Agent 1 (Place Recommender) via A2A.
    Returns a list of PlaceCandidate dicts.
    """
    # Agent 1 accepts free-form text (pydantic-ai based)
    parts = [f"Recommend places to visit in {city}."]
    if trip_reason:
        parts.append(f"Trip reason: {trip_reason}.")
    if preferences:
        parts.append(f"Preferences: {', '.join(preferences)}.")
    if budget is not None:
        parts.append(f"Activity budget: {budget} EUR.")
    parts.append(f"Trip dates: {trip_start} to {trip_end}.")

    prompt = " ".join(parts)

    task = await call_a2a_agent(
        base_url=settings.agent1_url,
        payload=prompt,
        max_wait=180.0,
    )

    # Agent 1 returns text — try to parse structured place data
    data = extract_data_artifact(task)
    if data and isinstance(data, dict):
        # Could be {"places": [...]} or {"result": [...]}
        for key in ("places", "place_candidates", "result"):
            if key in data and isinstance(data[key], list):
                return data[key]

    # Fall back to parsing text output as JSON
    text = extract_text_artifact(task)
    parsed = parse_json_from_text(text)
    if parsed:
        for key in ("places", "place_candidates", "result"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        if isinstance(parsed, list):
            return parsed

    # If all parsing fails, return raw text for logging
    print(f"⚠️  Agent 1 returned unparseable output: {text[:200]}")
    return []


async def call_clustering_agent(
    place_candidates: list[dict],
    trip_start: str,
    trip_end: str,
) -> list[list[dict]]:
    """
    Call Agent 2 (Clustering) via plain REST POST.
    Returns a list of clusters (each cluster is a list of PlaceCandidate dicts).
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.agent2_url}/cluster",
            json={
                "trip_start": trip_start,
                "trip_end": trip_end,
                "place_candidates": place_candidates,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data.get("clustered_place_candidates", [])


async def call_food_recommender(
    meal_slot: str,
    latitude: float,
    longitude: float,
    radius_meters: int = 1000,
    budget_per_person: float | None = None,
    preferences: list[str] | None = None,
) -> list[dict]:
    """
    Call Agent 4 (Food Recommender) via A2A.
    Returns a list of RestaurantCandidate dicts.
    """
    payload = json.dumps({
        "timeofday":              meal_slot,
        "searchcenter":           {"latitude": latitude, "longitude": longitude},
        "searchradiusmeters":     radius_meters,
        "budgetpermealperperson": budget_per_person,
        "preferences":            preferences,
    })

    task = await call_a2a_agent(
        base_url=settings.agent4_url,
        payload=payload,
        max_wait=60.0,
    )

    # Agent 4 returns DataPart with key "restaurantcandidates"
    data = extract_data_artifact(task, artifact_name="restaurant_candidates")
    if data and "restaurantcandidates" in data:
        return data["restaurantcandidates"]

    # Fall back to text parsing
    text = extract_text_artifact(task)
    parsed = parse_json_from_text(text)
    if parsed and "restaurantcandidates" in parsed:
        return parsed["restaurantcandidates"]

    print(f"⚠️  Agent 4 returned unparseable output: {text[:200]}")
    return []
