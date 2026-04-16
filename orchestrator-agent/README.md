# Agent 0 — Trip Planner Orchestrator

Main entry point of the trip planner multi-agent system.
Uses **OpenRouter** (OpenAI-compatible) as the LLM brain for parsing
natural language trip requests and generating itinerary summaries.

## Architecture

Receives a user trip request (free-text or structured JSON) and coordinates:
1. **Agent 1** (Place Recommender) — via A2A at `:8000`
2. **Agent 2** (Place Clustering) — via REST at `:8001`
3. **Agent 3** (Daily Scheduler) — built into the orchestrator
4. **Agent 4** (Food Recommender) — via A2A at `:8004`

## LLM Brain (OpenRouter)

The orchestrator uses OpenRouter to:
- **Parse free-text requests** → structured `TripRequest` JSON
  (e.g. "Plan me a 3-day trip to Milan with 600€ budget")
- **Generate summaries** of the final itinerary

Any OpenAI-compatible model via OpenRouter works. Default: `google/gemini-2.0-flash-001`.

## Quick Start

```bash
cp .env.example .env
# Fill in OPENROUTER_API_KEY

# Run locally
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Or with Docker
docker-compose up --build
```

## Test

```bash
# Unit tests (no external services needed)
pytest -m unit

# Integration tests (requires all agents running)
pytest -m integration

# Manual CLI test
python -m tests.test_client --scenario milan_3day
```
