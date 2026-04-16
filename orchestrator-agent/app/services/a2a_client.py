"""
A2A client for calling downstream agents (1, 2, 3).
All communication is via JSON-RPC message/send + tasks/get polling.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import httpx

from app.core.config import settings


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def _jsonrpc_body(method: str, params: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id":      str(uuid.uuid4()),
        "method":  method,
        "params":  params,
    }


async def send_a2a_message(
    base_url: str,
    payload: str,
    poll_interval: float = 1.5,
    max_wait: float = 120.0,
) -> dict:
    """
    Send a text message to any A2A agent, poll until completed,
    return the full task dict with artifacts.
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
        send_resp = await client.post(f"{base_url}/", json=send_body)
        send_resp.raise_for_status()
        task_id = send_resp.json().get("result", {}).get("id")
        if not task_id:
            raise RuntimeError(f"No task id from {base_url}: {send_resp.json()}")

        deadline = asyncio.get_event_loop().time() + max_wait
        while asyncio.get_event_loop().time() < deadline:
            poll_resp = await client.post(
                f"{base_url}/",
                json=_jsonrpc_body("tasks/get", {"id": task_id}),
                timeout=10,
            )
            poll_resp.raise_for_status()
            task = poll_resp.json().get("result", poll_resp.json())
            state = task.get("status", {}).get("state", "unknown")

            if state == "completed":
                return task
            if state in ("canceled", "failed"):
                raise RuntimeError(f"A2A task {task_id} at {base_url}: {state}")
            await asyncio.sleep(poll_interval)

    raise RuntimeError(f"A2A task {task_id} at {base_url} timed out ({max_wait}s)")


def extract_text_from_task(task: dict) -> str:
    """Pull the first text content from a completed A2A task."""
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                return part.get("text", "")
    # Fall back to history messages
    for msg in task.get("history", []):
        if msg.get("role") == "agent":
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    return part.get("text", "")
    return ""
