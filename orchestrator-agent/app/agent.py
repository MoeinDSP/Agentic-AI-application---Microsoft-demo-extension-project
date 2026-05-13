"""
LLM Agent Brain — OpenRouter (OpenAI-compatible) with tool calling.

The orchestrator is a conversational LLM that decides which A2A agents
to call based on the user's message. It can:
  - Chat naturally and ask clarifying questions
  - Call Agent 1 (recommend_places) to get place candidates
  - Call Agent 2 (cluster_places) to group them by day
  - Call Agent 3 (schedule_day) to build each day's itinerary
  - Summarise and present the final plan

The LLM drives the flow — there is no hardcoded pipeline.
"""
from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from app.core.config import settings
from app.services.a2a_client import extract_text_from_task, send_a2a_message


# ── OpenRouter client ─────────────────────────────────────────────────────────

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        headers = {}
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name
        _client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=headers or None,
        )
    return _client


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a friendly, expert trip-planning assistant.

You help users plan city trips by coordinating three specialist agents through tools:
1. **recommend_places** — finds attractions/places to visit in a city
2. **cluster_places** — groups those places into geographically coherent day-clusters
3. **schedule_day** — builds a chronological schedule for one day (including restaurants)

Workflow (you decide when each step is needed):
- First, understand what the user wants (city, dates, budget, preferences).
  Ask clarifying questions if anything important is missing.
- Call recommend_places to get a list of candidate places.
- Call cluster_places to group them into days.
- Call schedule_day once for each day-cluster.
- Present the final itinerary in a clear, readable format.

Budget rules (apply only when the user provides a budget):
- 70% of total budget goes to activities, 30% to food.
- Food budget is split evenly across trip days.
- Pass the per-day food budget to schedule_day.

Important:
- Always call the tools — never fabricate place names, ratings, or restaurants.
- You can call tools multiple times if the user asks to adjust the plan.
- Be conversational. If the user just says "hi", greet them and ask about their trip.
- Present itineraries with times, place names, and restaurant suggestions clearly.
"""


# ── Tool definitions (OpenAI function-calling schema) ─────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_places",
            "description": (
                "Get a list of recommended places to visit in a city. "
                "Returns place candidates with names, ratings, visit durations, and costs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Destination city name.",
                    },
                    "trip_start": {
                        "type": "string",
                        "description": "Trip start date/time in ISO 8601 format.",
                    },
                    "trip_end": {
                        "type": "string",
                        "description": "Trip end date/time in ISO 8601 format.",
                    },
                    "budget": {
                        "type": "number",
                        "description": "Activity budget in EUR (optional).",
                    },
                    "trip_reason": {
                        "type": "string",
                        "description": "Purpose of the trip (optional).",
                    },
                    "preferences": {
                        "type": "string",
                        "description": "Comma-separated interests, e.g. 'museums, architecture, italian food'.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_places",
            "description": (
                "Group a list of places into geographically coherent day-clusters. "
                "Each cluster represents one trip day. "
                "Input: the JSON array of place candidates + trip dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_candidates_json": {
                        "type": "string",
                        "description": "JSON string of the place candidates array from recommend_places.",
                    },
                    "trip_start": {
                        "type": "string",
                        "description": "Trip start date/time in ISO 8601.",
                    },
                    "trip_end": {
                        "type": "string",
                        "description": "Trip end date/time in ISO 8601.",
                    },
                },
                "required": ["place_candidates_json", "trip_start", "trip_end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_day",
            "description": (
                "Build a detailed chronological schedule for a single trip day. "
                "Takes a cluster of places and returns an ordered plan with visit times "
                "and restaurant recommendations for meals. "
                "Call this once per day-cluster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_json": {
                        "type": "string",
                        "description": "JSON string of places for this day (one cluster from cluster_places).",
                    },
                    "day_date": {
                        "type": "string",
                        "description": "The date for this day, e.g. '2026-06-10'.",
                    },
                    "food_budget_per_day": {
                        "type": "number",
                        "description": "Food budget in EUR for this day (optional).",
                    },
                    "preferences": {
                        "type": "string",
                        "description": "Comma-separated food/activity preferences (optional).",
                    },
                },
                "required": ["cluster_json", "day_date"],
            },
        },
    },
]


# ── Tool execution (A2A calls) ────────────────────────────────────────────────

async def _execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool by calling the corresponding downstream A2A agent.
    Returns the agent's text response to feed back to the LLM.
    """
    try:
        if name == "recommend_places":
            prompt_parts = [f"Recommend places to visit in {arguments['city']}."]
            if arguments.get("trip_reason"):
                prompt_parts.append(f"Trip reason: {arguments['trip_reason']}.")
            if arguments.get("preferences"):
                prompt_parts.append(f"Preferences: {arguments['preferences']}.")
            if arguments.get("budget"):
                prompt_parts.append(f"Activity budget: {arguments['budget']} EUR.")
            if arguments.get("trip_start"):
                prompt_parts.append(f"Dates: {arguments['trip_start']} to {arguments.get('trip_end', '')}.")

            task = await send_a2a_message(
                base_url=settings.agent1_url,
                payload=" ".join(prompt_parts),
                max_wait=180.0,
            )
            return extract_text_from_task(task)

        elif name == "cluster_places":
            payload = json.dumps({
                "place_candidates": json.loads(arguments["place_candidates_json"]),
                "trip_start": arguments["trip_start"],
                "trip_end": arguments["trip_end"],
            })
            task = await send_a2a_message(
                base_url=settings.agent2_url,
                payload=payload,
                max_wait=60.0,
            )
            return extract_text_from_task(task)

        elif name == "schedule_day":
            payload = json.dumps({
                "cluster": json.loads(arguments["cluster_json"]),
                "day_date": arguments["day_date"],
                "food_budget_per_day": arguments.get("food_budget_per_day"),
                "preferences": arguments.get("preferences"),
            })
            task = await send_a2a_message(
                base_url=settings.agent3_url,
                payload=payload,
                max_wait=180.0,
            )
            return extract_text_from_task(task)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": f"{name} failed: {type(e).__name__}: {e}"})


# ── Main agent loop ───────────────────────────────────────────────────────────

async def run_agent(
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Run the LLM tool-calling loop.

    Args:
        user_message: The latest user message.
        conversation_history: Prior messages (for multi-turn context).

    Returns:
        (final_text_response, updated_conversation_history)
    """
    client = _get_client()

    # Build messages
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Tool-calling loop
    for _round in range(settings.max_tool_rounds):
        response = await client.chat.completions.create(
            model=settings.openrouter_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=4096,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # Append assistant message to history
        msg_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        # If no tool calls — we have the final response
        if not assistant_msg.tool_calls:
            final_text = assistant_msg.content or ""
            # Return history without system prompt
            return final_text, messages[1:]

        # Execute each tool call
        for tc in assistant_msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"   🔧 Tool call: {tc.function.name}({list(args.keys())})")
            result = await _execute_tool(tc.function.name, args)
            print(f"   ✅ {tc.function.name} returned {len(result)} chars")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Safety: max rounds exceeded
    messages.append({
        "role": "assistant",
        "content": "I've reached the maximum number of steps. Here's what I have so far based on the information gathered.",
    })
    return messages[-1]["content"], messages[1:]
