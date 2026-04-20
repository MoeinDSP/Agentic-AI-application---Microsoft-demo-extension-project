# Agent 3

Agent 3 is a single-day scheduling service for the trip-planning architecture
described in the course PDF. It accepts a normalized `DaySchedulingRequest`,
builds a chronological day schedule, calls the route tool service for travel
estimates, and calls Agent 4 for restaurant-backed lunch and dinner events.

The implementation is assignment-MVP scoped: deterministic, local-first, and
ready to replace with real MCP/provider integrations later.

## API

- `GET /health`
- `GET /.well-known/agent-card.json`
- `POST /v1/plan`
- `POST /a2a`

`/v1/plan` and `/a2a` use the same scheduling contract. `/a2a` wraps the input
with `request_id` and `action: "plan_day"`.

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

## Example

```bash
curl -X POST http://127.0.0.1:8080/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Run

```bash
uv sync --group dev
uv run python -m uvicorn agent3.main:create_app --factory --reload --host 127.0.0.1 --port 8080
```

Local startup order:

1. Start `agent-3-mcp`.
2. Start `agent-4`.
3. Start `agent-3`.

## Test

```bash
uv run ruff check .
uv run pytest
```

## Environment Variables

- `AGENT3_MCP_BASE_URL`: route tool base URL.
- `AGENT3_AGENT4_BASE_URL`: Agent 4 base URL.
- `AGENT3_AGENT4_INVOCATION_MODE`: `http` or `a2a`.
- `AGENT3_FALLBACK_TRAVEL_MINUTES`: deterministic fallback route duration.
- `AGENT3_DEFAULT_TRANSPORT_MODE`: default public transport mode.
