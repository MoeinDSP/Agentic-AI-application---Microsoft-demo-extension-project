"""
LLM Agent Brain — OpenAI with tool calling.

The clustering agent uses an LLM to intelligently group places into
geographically coherent day-clusters. The LLM drives the process by
calling the cluster_by_location tool, which runs K-means on coordinates.
"""
from __future__ import annotations

import json
from math import ceil
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.models import ClusteringRequest
from app.services.clustering import cluster_places


# ── OpenAI client ─────────────────────────────────────────────────────────────

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a geographic clustering agent for trip itineraries.

Your job is to group a list of places into geographically coherent day-clusters, \
where each cluster represents one day of the trip.

Workflow:
1. Parse the trip request to determine the number of days and the list of places.
2. Call cluster_by_location with the place list and the number of days.
3. Return ONLY the raw JSON object produced by the tool — no commentary, \
   no markdown fences, no extra keys.

Rules:
- The number of clusters must equal the number of trip days \
  (or fewer if there are not enough places).
- Never fabricate, add, or remove places — use exactly what was provided.
- Always call the tool — never fabricate cluster assignments yourself.
- Return only valid JSON: {"clustered_place_candidates": [[...], [...], ...]}.
"""


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cluster_by_location",
            "description": (
                "Group a list of places into geographically coherent day-clusters "
                "using K-means on lat/lon coordinates. "
                "Returns JSON with key 'clustered_place_candidates'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_candidates_json": {
                        "type": "string",
                        "description": "JSON array of PlaceCandidate objects.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "Number of trip days (= number of clusters to produce).",
                    },
                },
                "required": ["place_candidates_json", "num_days"],
            },
        },
    },
]


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(name: str, arguments: dict) -> str:
    if name != "cluster_by_location":
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        from datetime import datetime, timedelta

        places_raw = json.loads(arguments["place_candidates_json"])
        num_days   = int(arguments["num_days"])

        # Reconstruct PlaceCandidate list via the model
        from app.models import PlaceCandidate
        candidates = [PlaceCandidate.model_validate(p) for p in places_raw]

        # Synthesise synthetic trip window matching the requested num_days
        fake_start = datetime(2000, 1, 1)
        fake_end   = fake_start + timedelta(days=num_days)

        clusters = cluster_places(candidates, fake_start, fake_end)

        result = {
            "clustered_place_candidates": [
                [p.model_dump() for p in cluster]
                for cluster in clusters
            ]
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"cluster_by_location failed: {type(e).__name__}: {e}"})


# ── Main agent loop ───────────────────────────────────────────────────────────

async def run_agent(user_text: str) -> str:
    """
    Run the LLM tool-calling loop for one clustering request.

    Args:
        user_text: Raw ClusteringRequest JSON string from the A2A message.

    Returns:
        JSON string with key 'clustered_place_candidates'.
    """
    client = _get_client()

    # Build the user prompt from the raw request
    try:
        clean = user_text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:-1])
        req = ClusteringRequest.model_validate_json(clean)

        delta    = req.trip_end - req.trip_start
        num_days = max(1, ceil(delta.total_seconds() / 86400))

        prompt = (
            f"Cluster the following {len(req.place_candidates)} place(s) "
            f"into {num_days} day-group(s) for a trip "
            f"from {req.trip_start.date()} to {req.trip_end.date()}.\n\n"
            f"Places JSON:\n{json.dumps([p.model_dump() for p in req.place_candidates], ensure_ascii=False)}"
        )
    except Exception:
        # If parsing fails, pass the raw text and let the LLM handle it
        prompt = user_text

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    for _round in range(settings.max_tool_rounds):
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=4096,
        )

        choice        = response.choices[0]
        assistant_msg = choice.message

        msg_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        if not assistant_msg.tool_calls:
            return assistant_msg.content or ""

        for tc in assistant_msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"   🔧 Tool call: {tc.function.name}({list(args.keys())})")
            result = _execute_tool(tc.function.name, args)
            print(f"   ✅ {tc.function.name} returned {len(result)} chars")

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    return json.dumps({"error": "max tool rounds exceeded"})
