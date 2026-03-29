# Agent 3

Agent 3 is a planning service that accepts a day plan request and returns a
deterministic MVP itinerary. The current implementation is local-first and
intentionally simple so the API contract can stabilize before richer planning
logic is added.

It also exposes a minimal A2A-compatible HTTP boundary for orchestrator
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
- `AGENT3_MCP_BASE_URL`: base URL for the independent `agent-3-mcp` tool service.
- `AGENT3_MCP_TIMEOUT_SECONDS`: short timeout for route-estimate HTTP calls.
- `AGENT3_FALLBACK_TRAVEL_MINUTES`: deterministic per-leg fallback when MCP route estimation fails.
- `AGENT3_DEFAULT_TRANSPORT_MODE`: default transport mode when no preference is provided.
- `AGENT3_LOG_LEVEL`: logging level.

## Architecture Notes

- `api/` owns HTTP routing only.
- `models/` owns request and response contracts.
- `services/` owns deterministic planner behavior.
- `core/` owns config and logging.

## Planner MVP Behavior

The current planner uses a deterministic greedy algorithm:

- Sort places by `priority` descending.
- Preserve input order when priorities tie.
- Start scheduling at `day_start`.
- Resolve a single effective transport mode from `transport_preferences`.
- Supported modes are `walk`, `drive`, and `transit`.
- Default to `walk` when no transport preference is provided.
- Request travel time over HTTP from `agent-3-mcp` for:
  - start location -> first stop
  - previous accepted stop -> next candidate stop
- Treat places without `opens_at` and `closes_at` as unrestricted.
- Drop a place with `closed_at_arrival` when arrival happens before opening or at/after closing.
- Drop a place with `closes_before_visit_ends` when the visit would run past closing.
- Do not wait for a place to open. The planner either schedules it immediately or drops it.
- If `lunch_required` is `true`, attempt to insert exactly one synthetic lunch stop.
- Lunch is inserted at the earliest feasible point inside the configured lunch window.
- Lunch uses `lunch_time_window_start`, `lunch_time_window_end`, and `lunch_duration_minutes`.
- If lunch cannot fit, the planner keeps the place plan and adds `lunch_not_inserted` to `notes`.
- Add places one by one while travel time plus visit duration still finishes on or before `day_end`.
- Drop any place that no longer fits with reason `insufficient_time`.
- If MCP route estimation fails, use `AGENT3_FALLBACK_TRAVEL_MINUTES` for that leg and note the fallback in the response.

Response semantics:

- `ordered_stops` includes scheduled places with `arrival_time`, `start_time`, `end_time`, and `travel_minutes_from_previous`.
- Lunch appears in `ordered_stops` as a synthetic stop with `stop_type` set to `lunch`.
- `dropped_places` includes unscheduled places and a drop reason.
- `feasibility` is `true` when at least one stop was scheduled.
- `feasibility` is `false` when the planner cannot schedule any stop in the day window.
- `selected_transport_mode` shows the effective mode used for planning.
- `total_travel_minutes` is the sum of accepted travel legs.
- `total_visit_minutes` is the sum of accepted visit durations.

## A2A Support

Current scope:

- `GET /.well-known/agent-card.json` exposes discovery metadata.
- `POST /a2a` accepts a minimal typed A2A-shaped planning request.
- `POST /v1/plan` remains the direct domain API and is still preserved.

Current limitations:

- This is not a full A2A protocol implementation.
- Authentication is still a local-first placeholder.
- The planner is deterministic but still intentionally simple.
- Travel time is route-aware, but the planner is still greedy and not globally optimized.
- Only one effective transport mode is used for the whole plan.
- Lunch is synthetic only and does not include restaurant selection.

## Example Requests

Health:

```bash
curl http://127.0.0.1:8080/health
```

Direct planning request:

```bash
curl -X POST http://127.0.0.1:8080/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
    "start_location": {"lat": 41.9028, "lng": 12.4964, "name": "Rome"},
    "day_start": "09:00:00",
    "day_end": "15:00:00",
    "transport_preferences": ["walk"],
    "lunch_required": true,
    "lunch_time_window_start": "12:00:00",
    "lunch_time_window_end": "14:00:00",
    "lunch_duration_minutes": 30,
    "places": [
      {
        "id": "colosseum",
        "name": "Colosseum",
        "lat": 41.8902,
        "lng": 12.4922,
        "estimated_duration_minutes": 180,
        "priority": 5,
        "opens_at": "09:00:00",
        "closes_at": "18:00:00"
      },
      {
        "id": "pantheon",
        "name": "Pantheon",
        "lat": 41.8986,
        "lng": 12.4769,
        "estimated_duration_minutes": 45,
        "priority": 4,
        "opens_at": "10:30:00",
        "closes_at": "18:00:00"
      }
    ]
  }'
```

Travel-aware response shape:

```json
{
  "ordered_stops": [
    {
      "stop_type": "place",
      "place_id": "colosseum",
      "place_name": "Colosseum",
      "sequence": 1,
      "arrival_time": "09:10:00",
      "start_time": "09:10:00",
      "end_time": "12:10:00",
      "travel_minutes_from_previous": 10,
      "estimated_duration_minutes": 180
    },
    {
      "stop_type": "lunch",
      "place_id": "lunch",
      "place_name": "Lunch",
      "sequence": 2,
      "arrival_time": "12:10:00",
      "start_time": "12:10:00",
      "end_time": "12:40:00",
      "travel_minutes_from_previous": 0,
      "estimated_duration_minutes": 30
    }
  ],
  "dropped_places": [
    {
      "place_id": "pantheon",
      "reason": "insufficient_time"
    }
  ],
  "notes": [
    "deterministic_greedy_planner",
    "travel_time_source=mcp",
    "feasibility=true_when_at_least_one_stop_is_scheduled",
    "selected_transport_mode=walk",
    "transport_preferences=walk",
    "lunch_inserted"
  ],
  "feasibility": true,
  "selected_transport_mode": "walk",
  "total_travel_minutes": 10,
  "total_visit_minutes": 210
}
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
      "day_end": "15:00:00",
      "transport_preferences": ["walk"],
      "lunch_required": true,
      "lunch_time_window_start": "12:00:00",
      "lunch_time_window_end": "14:00:00",
      "lunch_duration_minutes": 30,
      "places": [
        {
          "id": "colosseum",
          "name": "Colosseum",
          "lat": 41.8902,
          "lng": 12.4922,
          "estimated_duration_minutes": 180,
          "priority": 5,
          "opens_at": "09:00:00",
          "closes_at": "18:00:00"
        }
      ]
    }
  }'
```

## A2A Flow

```text
orchestrator
  -> agent-3 A2A boundary
  -> deterministic planner MVP
  -> future tool service integrations
```

## Developer Workflow

Standard targets:

- `install`
- `run`
- `test`
- `lint`

Local startup order:

1. Start `agent-3-mcp`
2. Start `agent-3`

Example local commands:

```bash
cd services/agent-3-mcp
uv sync --group dev
uv run python -m uvicorn agent3_mcp.main:app --factory --reload --host 127.0.0.1 --port 8090
```

```bash
cd services/agent-3
uv sync --group dev
uv run python -m uvicorn agent3.main:app --factory --reload --host 127.0.0.1 --port 8080
```

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
- Opening hours are supported only as simple same-day place windows with no waiting logic.
- Lunch is supported only as one synthetic stop inside a configured time window.
- There is no restaurant recommendation or external lunch search yet.
- Dinner is not modeled yet.
- Advanced transport-mode optimization is not modeled yet.
- Global route optimization is not modeled yet.
- Multi-modal plans are not modeled yet.
- Global lunch placement optimization is not modeled yet.
- Richer constraints should be added next, such as hard stop ordering rules and waiting behavior.
- Future tool calls should flow into `agent-3-mcp` instead of being embedded
  directly into the agent boundary.
