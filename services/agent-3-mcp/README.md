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

Route requests can also include `departure_time`, which Agent 3 forwards from
the actual travel start time. This is especially important for transit
estimates.

## Configuration

Copy `.env.example` to `.env` if needed.

All settings use the `AGENT3_MCP_` prefix.

| Variable | Required | Source | Meaning |
| --- | --- | --- | --- |
| `AGENT3_MCP_APP_NAME` | No | local/deploy | Service name override. |
| `AGENT3_MCP_ENVIRONMENT` | No | local/deploy | Environment label such as `development` or `production`. |
| `AGENT3_MCP_HOST` | No | local | Local bind host. |
| `AGENT3_MCP_PORT` | No | local | Local bind port. |
| `AGENT3_MCP_GOOGLE_MAPS_API_KEY` | Yes | local/deploy | Google Maps Platform key used for Routes API requests. In production this is injected from Secret Manager. |
| `AGENT3_MCP_GOOGLE_ROUTES_TIMEOUT_SECONDS` | No | local/deploy | HTTP timeout for Google Routes requests. |
| `AGENT3_MCP_LOG_LEVEL` | No | local/deploy | Application log level. |

Local startup typically sets:

- `AGENT3_MCP_GOOGLE_MAPS_API_KEY=<real Google Maps key>`
- `AGENT3_MCP_GOOGLE_ROUTES_TIMEOUT_SECONDS=5.0`

Production deployment sets:

- `AGENT3_MCP_ENVIRONMENT=production`
- `AGENT3_MCP_GOOGLE_ROUTES_TIMEOUT_SECONDS=5.0`
- `AGENT3_MCP_GOOGLE_MAPS_API_KEY=<Secret Manager injected secret>`

Production notes:

- the Secret Manager value must be non-empty and contain a real Google Maps key
- surrounding secret whitespace is stripped by MCP before the key is used
- deploy smoke uses a minimal walking route-estimate request without
  `departure_time`

For private Cloud Run deployment, MCP relies on Cloud Run IAM for ingress
protection. Agent 3 must call it with a Google-signed ID token and a runtime
service account that has `roles/run.invoker` on the MCP service.

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
- Agent 3 and Agent 3 MCP are the only repo-owned deployable services today.
- Agent 3 MCP is deployed from this repo. Agent 4 is not.
- `place-details` is intentionally still placeholder in this pass.
- Google Routes HTTP error bodies are now logged to aid production debugging.
- Route estimation can still fail on some hops and trigger Agent 3 fallback
  behavior even when the overall system remains healthy.
- For production deployment guidance, see
  [../../docs/gcp-cloud-run.md](../../docs/gcp-cloud-run.md).
