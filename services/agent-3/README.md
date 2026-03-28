# Agent 3

Agent 3 is a planning service that accepts a day plan request and returns a
deterministic placeholder itinerary. The current implementation is local-first
and intentionally simple so the API contract can stabilize before real planning
logic is added.

It now also exposes a minimal A2A-compatible HTTP boundary for orchestrator
discovery and invocation. This is A2A-shaped and A2A-ready, not a full
spec-complete A2A implementation.

## Purpose

- Expose a typed HTTP API for itinerary planning.
- Expose a minimal A2A-ready discovery and invocation boundary for an
  orchestrator.
- Provide a production-sensible baseline for local development and later Cloud
  Run deployment.

## Structure

```text
src/agent3/
  api/
  core/
  models/
  services/
tests/
```

## Prerequisites

- Python 3.11+
- `uv`

## Install

```bash
uv sync
```

Or, if `make` is available:

```bash
make install
```

## Run

```bash
uv run python -m uvicorn agent3.main:app --factory --reload --host 127.0.0.1 --port 8080
```

Or:

```bash
make run
```

## Test

```bash
uv run pytest
```

Or:

```bash
make test
```

## Lint

```bash
uv run ruff check .
```

Or:

```bash
make lint
```

## Environment Variables

Copy `.env.example` to `.env` if needed.

- `AGENT3_APP_NAME`: FastAPI application name.
- `AGENT3_ENVIRONMENT`: environment label for logs and config.
- `AGENT3_SERVICE_VERSION`: version surfaced in the Agent Card.
- `AGENT3_HOST`: bind host for local runs.
- `AGENT3_PORT`: bind port for local runs and Cloud Run compatibility.
- `AGENT3_PUBLIC_BASE_URL`: public base URL used in the Agent Card.
- `AGENT3_AGENT_CARD_PATH`: Agent Card path.
- `AGENT3_A2A_PATH`: A2A request path.
- `AGENT3_LOG_LEVEL`: logging level.

## Architecture Notes

- `api/` owns HTTP routing only.
- `models/` owns request and response contracts.
- `services/` owns placeholder planning behavior.
- `core/` owns config and logging.

## A2A Support

Current scope:

- `GET /.well-known/agent-card.json` exposes discovery metadata.
- `POST /a2a` accepts a minimal typed A2A-shaped planning request.
- `POST /v1/plan` remains the direct domain API and is still preserved.

Current limitations:

- This is not a full A2A protocol implementation.
- Authentication is still a local-first placeholder.
- Planner behavior is still deterministic stub logic.

## Example Requests

Health:

```bash
curl http://127.0.0.1:8080/health
```

Agent Card:

```bash
curl http://127.0.0.1:8080/.well-known/agent-card.json
```

Minimal A2A request:

```bash
curl -X POST http://127.0.0.1:8080/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-123",
    "action": "plan_day",
    "input": {
      "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
      "day_start": "09:00:00",
      "day_end": "11:00:00",
      "transport_preferences": ["walk"],
      "places": [
        {
          "id": "colosseum",
          "name": "Colosseum",
          "lat": 41.8902,
          "lng": 12.4922,
          "estimated_duration_minutes": 90,
          "priority": 5
        }
      ]
    }
  }'
```

## A2A Flow

```text
orchestrator
  -> agent-3 A2A boundary
  -> planner stub
  -> future tool service integrations
```

## Developer Workflow

Standard targets:

- `install`
- `run`
- `test`
- `lint`

## Docker Usage

Build the image from the service directory:

```bash
docker build -t agent-3:local .
```

Run the container locally:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 agent-3:local
```

## Future Integration Notes

- Real A2A protocol features can replace the current HTTP/JSON adapter without a
  full rewrite of the planning API.
- Real planner logic should replace the deterministic placeholder service
  without changing the API surface.
- Future tool calls should flow into `agent-3-mcp` instead of being embedded
  directly into the agent boundary.
