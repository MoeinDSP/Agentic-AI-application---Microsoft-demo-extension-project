# Agent 0 — Trip Planner Orchestrator

Main entry point of the trip-planner multi-agent system.

This service is a FastA2A agent that uses **OpenRouter** (OpenAI-compatible API)
to run an LLM tool-calling loop and coordinate downstream agents.

## What this agent does

Given a natural-language trip request (for example: *"Plan me a 3-day trip to Milan with a 600€ budget"*), the orchestrator:

1. Calls **Agent 1** (`recommend_places`) to fetch candidate places.
2. Calls **Agent 2** (`cluster_places`) to split places into day-clusters.
3. Calls **Agent 3** (`schedule_day`) to build day-by-day itineraries.

The flow is LLM-driven (not hardcoded), and the final response is returned as an A2A task artifact.

## Architecture

- **This service (Agent 0)**: FastA2A server on `:8080`
- **Agent 1 (Place Recommender)**: A2A at `http://localhost:8000`
- **Agent 2 (Place Clustering)**: A2A at `http://localhost:8001`
- **Agent 3 (Daily Scheduler)**: A2A at `http://localhost:8003`

> Note: Food suggestions are handled downstream by the scheduler pipeline (Agent 3 side).

## LLM configuration (OpenRouter)

The orchestrator uses any OpenAI-compatible model via OpenRouter.

- Default model: `google/gemini-2.0-flash-001`
- Base URL default: `https://openrouter.ai/api/v1`

Required environment variable:

- `OPENROUTER_API_KEY`

## Setup

### 1) Configure environment

```bash
cp .env.example .env
```

Then set your key in `.env`:

- `OPENROUTER_API_KEY=...`

You can also override these if needed:

- `OPENROUTER_MODEL`
- `AGENT1_URL`
- `AGENT2_URL`
- `AGENT3_URL`

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Agent card endpoint:

- `GET /.well-known/agent.json`

## Run with Docker

```bash
docker-compose up --build
```

This exposes container port `8080` (configurable via `AGENT_PORT`).

## Testing

Current test suite in `tests/test_orchestrator_mocked.py` is **unit-level** and uses mocked downstream agents/LLM.

Run tests from `orchestrator-agent/`:

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Exclude integration marker (if/when added)
pytest -m "not integration"
```

`pytest.ini` defines both `unit` and `integration` markers.
At the moment, the repository contains mocked unit tests.
