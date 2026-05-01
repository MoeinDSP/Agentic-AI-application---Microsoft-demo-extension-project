# Agent 3 MCP

Agent 3 MCP is the tool service used by Agent 3 for routing and auxiliary place
lookups. The route estimate endpoint now uses the Google Maps Routes API. The
place-details endpoint remains placeholder-only in this pass.

## API

- `GET /health`
- `POST /v1/tools/route-estimate`
- `POST /v1/tools/place-details`

## Route Provider

`POST /v1/tools/route-estimate` calls the Google Routes `computeRoutes` REST
method and maps the result back into the existing MCP response shape:

- `mode`
- `estimated_distance_km`
- `estimated_duration_minutes`
- `notes`

Supported public transport modes:

- `walking`
- `driving`
- `transit`
- `bicycling`

## Configuration

Copy `.env.example` to `.env` if needed.

- `AGENT3_MCP_GOOGLE_MAPS_API_KEY`: required for Google Routes requests.
- `AGENT3_MCP_GOOGLE_ROUTES_TIMEOUT_SECONDS`: provider HTTP timeout.

## Run

```bash
uv sync --group dev
uv run python -m uvicorn agent3_mcp.main:create_app --factory --reload --host 127.0.0.1 --port 8090
```

## Test

```bash
uv run ruff check .
uv run pytest
```

## Notes

- Route requests use coordinate-only inputs and request duration and distance.
- `place-details` is intentionally still placeholder in this pass.
- For production deployment guidance, see
  [../../docs/gcp-cloud-run.md](../../docs/gcp-cloud-run.md).
