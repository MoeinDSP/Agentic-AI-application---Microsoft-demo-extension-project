# Agent 3 MCP

Agent 3 MCP is an independent HTTP tool service for the Agent 3 planning
domain. It currently exposes deterministic placeholder endpoints for route
estimation and place details so the service boundary can stabilize before real
external integrations are added.

## Purpose

- Keep tool access separate from Agent 3 business logic.
- Provide a clean HTTP baseline that can later be wrapped for richer MCP
  protocol support.
- Stay local-first while remaining deployable to Cloud Run later.

## Structure

```text
src/agent3_mcp/
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
uv sync --group dev
```

## Run

```bash
uv run uvicorn agent3_mcp.main:app --factory --reload --host 127.0.0.1 --port 8090
```

## Test

```bash
uv run pytest
```

## Lint

```bash
uv run ruff check .
```

## Environment Variables

Copy `.env.example` to `.env` if needed.

- `AGENT3_MCP_APP_NAME`: FastAPI application name.
- `AGENT3_MCP_ENVIRONMENT`: environment label for logs and config.
- `AGENT3_MCP_HOST`: bind host for local runs.
- `AGENT3_MCP_PORT`: bind port for local runs and Cloud Run compatibility.
- `AGENT3_MCP_LOG_LEVEL`: logging level.

## Docker Usage

Docker support will be added in a later scaffold step.

## Future Integration Notes

- Real route and place providers should replace the deterministic placeholders.
- The current HTTP tool design is intentionally narrow so it can be wrapped or
  upgraded to MCP later without redesigning the service boundary.
