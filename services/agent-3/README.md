# Agent 3

Agent 3 is a planning service that accepts a day plan request and returns a
deterministic placeholder itinerary. The current implementation is local-first
and intentionally simple so the API contract can stabilize before real planning
logic is added.

## Purpose

- Expose a typed HTTP API for itinerary planning.
- Keep the service ready for future A2A exposure without coupling that work into
  the initial scaffold.
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
uv run uvicorn agent3.main:app --factory --reload --host 127.0.0.1 --port 8080
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
- `AGENT3_HOST`: bind host for local runs.
- `AGENT3_PORT`: bind port for local runs and Cloud Run compatibility.
- `AGENT3_LOG_LEVEL`: logging level.

## Architecture Notes

- `api/` owns HTTP routing only.
- `models/` owns request and response contracts.
- `services/` owns placeholder planning behavior.
- `core/` owns config and logging.

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

- The service is designed to become an HTTP-facing A2A boundary later.
- Real planner logic should replace the deterministic placeholder service
  without changing the API surface.
