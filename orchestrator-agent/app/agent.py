"""
LLM Brain — OpenRouter (OpenAI-compatible).

Uses the OpenAI SDK pointed at OpenRouter to:
  1. Parse free-text user requests into structured TripRequest JSON
  2. (future) Make scheduling decisions, generate summaries, etc.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from openai import AsyncOpenAI

from app.core.config import settings

# ── OpenRouter client (OpenAI-compatible) ─────────────────────────────────────

_client: AsyncOpenAI | None = None


def get_llm_client() -> AsyncOpenAI:
    """Lazy-init the OpenAI client pointed at OpenRouter."""
    global _client
    if _client is None:
        extra_headers = {}
        if settings.openrouter_site_url:
            extra_headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            extra_headers["X-Title"] = settings.openrouter_app_name

        _client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=extra_headers or None,
        )
    return _client


# ── System prompt for trip-request extraction ─────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are a trip-request parser. The user will describe a trip in natural language.
Extract the structured information and return ONLY a valid JSON object — no markdown,
no explanation, no extra text.

Today's date is {today}.

JSON schema to follow:
{{
  "city": "<string — destination city>",
  "trip_start": "<ISO 8601 datetime — e.g. 2026-06-10T09:00:00>",
  "trip_end": "<ISO 8601 datetime>",
  "location": {{"latitude": <float>, "longitude": <float>}} | null,
  "budget": {{"total_budget": <float>, "currency": "EUR"}} | null,
  "trip_reason": "<string>" | null,
  "preferences": ["<string>", ...] | null
}}

Rules:
- If the user specifies dates, use them exactly.
- If the user says "3 days" without dates, start from tomorrow.
- If the user doesn't mention a budget, set budget to null.
- If the user doesn't mention accommodation location, set location to null.
- Preferences include food types, activity types, interests, etc.
- Always output valid JSON. Nothing else.
"""


async def parse_user_request(user_text: str) -> dict | None:
    """
    Use the LLM to extract a structured TripRequest from free-text.
    Returns a dict matching the TripRequest schema, or None on failure.
    """
    client = get_llm_client()
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        response = await client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": _EXTRACTION_SYSTEM_PROMPT.format(today=today),
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""

        # Strip markdown fences if model still wraps them
        clean = content.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:-1])

        return json.loads(clean)

    except Exception as e:
        print(f"⚠️  LLM parsing failed: {type(e).__name__}: {e}")
        return None


# ── Summary generation (post-pipeline) ────────────────────────────────────────

_SUMMARY_SYSTEM_PROMPT = """\
You are a friendly travel assistant. Given a trip itinerary in JSON format,
write a short, warm, 3-4 sentence summary of the trip highlighting the main
activities and restaurants. Keep it casual and enthusiastic.
Do NOT include JSON in your response — just the plain text summary.
"""


async def generate_itinerary_summary(itinerary_json: str) -> str:
    """Generate a human-friendly summary of the itinerary."""
    client = get_llm_client()

    try:
        response = await client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": itinerary_json},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        print(f"⚠️  Summary generation failed: {e}")
        return ""
