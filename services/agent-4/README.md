# Agent 4

Agent 4 is a local-first mock food recommender service. It exposes a typed HTTP
API for deterministic meal recommendations so Agent 3 can enrich a scheduled
lunch slot without owning restaurant search logic.

## Purpose

- Expose a typed meal recommendation API.
- Keep restaurant recommendation logic independent from Agent 3.
- Provide a deterministic mock service that can later be replaced with a real
  backend.

## Structure

- `src/agent4/api/` owns HTTP routing.
- `src/agent4/models/` owns request and response contracts.
- `src/agent4/services/` owns deterministic recommendation behavior.
- `src/agent4/core/` owns config and logging.
- `tests/` covers health, validation, and recommendation behavior.

## Prerequisites

- Python 3.11+
- `uv`

## Install

```bash
uv sync --group dev
```

## Run

```bash
uv run python -m uvicorn agent4.main:app --factory --reload --host 127.0.0.1 --port 8070
```

Local development order when used with Agent 3:

1. start `agent-3-mcp`
2. start `agent-4`
3. start `agent-3`

## Test

```bash
uv run pytest
```

## Lint

```bash
uv run ruff check .
```

## Environment Variables

- `AGENT4_APP_NAME`
- `AGENT4_ENVIRONMENT`
- `AGENT4_SERVICE_VERSION`
- `AGENT4_HOST`
- `AGENT4_PORT`
- `AGENT4_LOG_LEVEL`

## API

- `GET /health`
- `POST /v1/recommend-meal`

Current mock request fields:

- `time_of_day`
- `search_center`
- `search_radius_meters`
- `budget_per_meal_per_person`
- `preferences`

Current mock response fields:

- `candidates`

Deterministic ranking inputs:

- budget fit
- cuisine or preference keyword match
- rating as tie-breaker
- distance within `search_radius_meters`

## Docker

```bash
docker build -t agent-4:local .
docker run --rm -p 8070:8070 -e PORT=8070 agent-4:local
```

## Notes

- This service is deterministic and mocked.
- No external restaurant APIs are used yet.
- Lunch is the primary supported meal type for now.
- Agent 3 remains the scheduler. Agent 4 only returns ranked meal candidates.
