# Agent 3

Agent 3 is a single-day scheduling service for the trip-planning architecture
described in the course PDF. It accepts a normalized `DaySchedulingRequest`,
builds a chronological day schedule, calls the route tool service for travel
estimates, and calls the real external Agent 4 for restaurant-backed lunch and
dinner events.

The scheduler contract stays stable in this pass, but the integrations now
target a real external A2A food recommender and a real Google Routes-backed MCP
service.

## API

- `GET /health`
- `GET /.well-known/agent-card.json`
- `POST /`

`POST /` is a FastA2A JSON-RPC endpoint. The current implementation supports:

- `message/send`
- `tasks/get`
- `tasks/cancel`

`message/send` should include the scheduling payload in
`params.message.parts[0].data` with `kind: "data"`.

## Day Scheduling Contract

Request fields:

- `places`: list of candidate places.
- `day_start`: schedule start datetime.
- `day_end`: schedule end datetime.
- `food_budget_per_day`: optional total food budget for the day.
- `preferences`: optional food/place preference terms.
- `acceptable_transport_modes`: one or more of `walking`, `driving`, `transit`,
  or `bicycling`.

Place fields:

- `id`
- `name`
- `location`: `latitude`, `longitude`, and optional `address`.
- `estimated_visit_duration_minutes`
- `estimated_cost`
- `category`
- `rating`
- `summary`
- `priority_score`
- `opening_hours`: entries with `day_of_week`, `open_time`, and `close_time`.

Response fields:

- `day_schedule.date`
- `day_schedule.events`
- `unscheduled_places`
- `warnings`

Schedule events are chronological and use `event_type`:

- `travel`
- `visit`
- `meal`

## Planner Behavior

- Sorts candidate places by `priority_score` descending.
- Preserves input order when priorities tie.
- Schedules visits inside `day_start` and `day_end`.
- Uses same-day opening hours when available.
- Adds infeasible places to `unscheduled_places`.
- Emits warning strings for dropped places, route fallback, and meal fallback.
- Does not add inbound travel before the first scheduled event because the PDF
  request shape has no start location.
- Adds travel events between scheduled places and meals.

## Meals

- Lunch is attempted when the day overlaps `12:00-14:00`.
- Dinner is attempted when the day overlaps `19:00-21:00`.
- Meal duration is fixed at 60 minutes for the assignment MVP.
- `food_budget_per_day` is split across the meal slots Agent 3 attempts.
- Agent 4 is called for each inserted lunch or dinner.
- If Agent 4 is unavailable or returns no candidates, Agent 3 inserts a
  synthetic meal event and records a warning.
- If a meal window has not started yet, Agent 3 delays restaurant travel until
  it is actually needed instead of sending the user to the restaurant early.
- Agent 4 is treated as an external dependency and is configured through
  `AGENT3_AGENT4_BASE_URL`.

## Example

```bash
curl -X POST http://127.0.0.1:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "schedule-1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "message-1",
        "role": "user",
        "kind": "message",
        "parts": [
          {
            "kind": "data",
            "data": {
              "day_start": "2026-04-20T09:00:00",
              "day_end": "2026-04-20T22:00:00",
              "food_budget_per_day": 40,
              "preferences": ["italian"],
              "acceptable_transport_modes": ["walking", "bicycling"],
              "places": [
                {
                  "id": "colosseum",
                  "name": "Colosseum",
                  "location": {"latitude": 41.8902, "longitude": 12.4922},
                  "estimated_visit_duration_minutes": 180,
                  "estimated_cost": 18,
                  "category": "historical",
                  "rating": 4.9,
                  "summary": "Roman amphitheatre.",
                  "priority_score": 5,
                  "opening_hours": [
                    {
                      "day_of_week": "monday",
                      "open_time": "09:00:00",
                      "close_time": "18:00:00"
                    }
                  ]
                },
                {
                  "id": "pantheon",
                  "name": "Pantheon",
                  "location": {"latitude": 41.8986, "longitude": 12.4769},
                  "estimated_visit_duration_minutes": 45,
                  "priority_score": 4
                }
              ]
            }
          }
        ]
      },
      "configuration": {
        "acceptedOutputModes": ["application/json"]
      }
    }
  }'
```

## Run

```bash
uv sync --group dev
uv run python -m uvicorn agent3.main:create_app --factory --reload --host 127.0.0.1 --port 8080
```

Local startup order:

1. Start `agent-3-mcp`.
2. Set `AGENT3_AGENT4_BASE_URL` to the external Agent 4 endpoint you want to use.
3. Start `agent-3`.

## Test

```bash
uv run ruff check .
uv run pytest
```

## Environment Variables

All settings use the `AGENT3_` prefix.

| Variable | Required | Source | Meaning |
| --- | --- | --- | --- |
| `AGENT3_APP_NAME` | No | local/deploy | Service name override. |
| `AGENT3_ENVIRONMENT` | No | local/deploy | Environment label such as `development` or `production`. |
| `AGENT3_SERVICE_VERSION` | No | local/deploy | Version string exposed by the FastA2A agent. |
| `AGENT3_HOST` | No | local | Local bind host. |
| `AGENT3_PORT` | No | local | Local bind port. |
| `AGENT3_PUBLIC_BASE_URL` | Yes in deploy, no locally | local/deploy | Public URL advertised in the FastA2A agent card. Cloud Run deployment injects the real service URL automatically. |
| `AGENT3_MCP_BASE_URL` | Yes | local/deploy | Base URL for the Agent 3 MCP route tool service. |
| `AGENT3_MCP_TIMEOUT_SECONDS` | No | local/deploy | HTTP timeout for Agent 3 to MCP requests. |
| `AGENT3_AGENT4_BASE_URL` | Yes for production meals | local/deploy | External Agent 4 base URL. Agent 4 is not implemented in this repo. |
| `AGENT3_AGENT4_TIMEOUT_SECONDS` | No | local/deploy | HTTP timeout for Agent 4 calls. |
| `AGENT3_AGENT4_INVOCATION_MODE` | No | local/deploy | Agent 4 invocation mode. Current production setting is `a2a`. |
| `AGENT3_AGENT4_POLL_INTERVAL_SECONDS` | No | local/deploy | Poll interval for FastA2A task completion when calling Agent 4. |
| `AGENT3_AGENT4_MAX_WAIT_SECONDS` | No | local/deploy | Maximum wait for Agent 4 FastA2A task completion. |
| `AGENT3_FALLBACK_TRAVEL_MINUTES` | No | local/deploy | Deterministic travel duration used when MCP route estimation fails. |
| `AGENT3_DEFAULT_TRANSPORT_MODE` | No | local/deploy | Default transport mode when none is supplied in the request. |
| `AGENT3_LOG_LEVEL` | No | local/deploy | Application log level. |

Local startup typically sets:

- `AGENT3_PUBLIC_BASE_URL=http://127.0.0.1:8080`
- `AGENT3_MCP_BASE_URL=http://127.0.0.1:8090`
- `AGENT3_AGENT4_BASE_URL=<external Agent 4 URL>`

Production deployment sets:

- `AGENT3_ENVIRONMENT=production`
- `AGENT3_PUBLIC_BASE_URL=<deployed Cloud Run URL>`
- `AGENT3_MCP_BASE_URL=<deployed agent-3-mcp URL>`
- `AGENT3_AGENT4_BASE_URL=<external Agent 4 URL from GitHub variables>`

Agent 3 does not own or deploy Agent 4. It only consumes an external URL.

## Deployment

Manual Cloud Run deployment steps and Google Secret Manager wiring are
documented in [../../docs/gcp-cloud-run.md](../../docs/gcp-cloud-run.md).
