# Trip Planner — Agentic AI Multi-Agent System

> A Microsoft-demo extension project built as part of a cloud computing course at Politecnico di Milano. It implements a fully automated, AI-driven trip planning system using a multi-agent architecture where several independent AI services collaborate to produce a complete day-by-day travel itinerary from a single natural-language request.

---

## Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [How It Works — The Big Picture](#how-it-works--the-big-picture)
3. [System Architecture](#system-architecture)
4. [Agent-by-Agent Breakdown](#agent-by-agent-breakdown)
5. [Technology Stack](#technology-stack)
6. [Repository Structure](#repository-structure)
7. [Running Locally](#running-locally)
8. [Cloud Deployment](#cloud-deployment)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Environment Variables Reference](#environment-variables-reference)
11. [Testing](#testing)
12. [Contributors](#contributors)

---

## What Is This Project?

This project is an **agentic AI application** — a system where multiple AI "agents" (independent services, each with its own LLM, tools, and responsibilities) work together to answer a single user question.

The user asks something like:

> *"Plan me a 3-day trip to Milan with a 600€ budget."*

The system automatically breaks this request down, passes it through a chain of specialized agents, and returns a fully structured day-by-day itinerary — complete with visiting places, meal recommendations, travel time estimates, opening hours, and budget tracking — all without any manual steps.

---

## How It Works — The Big Picture

When a user sends a natural-language trip request, the following pipeline executes automatically:

1. **The Orchestrator (Agent 0)** receives the request, uses an LLM to understand it, and decides which downstream agents to call and in what order.
2. **The Place Recommender (Agent 1)** searches for tourist and cultural places in the destination city that match the user's preferences and budget, using the Google Maps MCP tool.
3. **The Place Clusterer (Agent 2)** takes the list of recommended places and groups them into day-clusters — deciding which places to visit together on the same day based on proximity and logical grouping.
4. **The Day Scheduler (Agent 3)** takes each day's cluster of places and builds a precise, chronologically ordered schedule — adding real travel time between places (via the Google Maps Routes API), inserting meal breaks, and flagging places that cannot fit in the time window.
5. **The Food Recommender (Agent 4)** is called by Agent 3 to suggest real restaurants for lunch and dinner slots. Agent 4 is an external service and is not deployed from this repository.

The final result is a complete multi-day itinerary returned to the user as a structured A2A task artifact.

---

## System Architecture

```
User Request (natural language)
        |
        v
+----------------------------------------------+
|   Agent 0 - Orchestrator                     |
|   FastA2A, LLM tool-calling loop             |
|   Port :8080  |  OpenRouter / Gemini          |
+----------------------------------------------+
        |  A2A JSON-RPC calls
   +----+--------------------+
   v                         v
Agent 1                   Agent 2
Place Recommender         Place Clusterer
:8000 | pydantic-ai       :8001 | pydantic-ai
Google Maps MCP           Azure OpenAI (gpt-4o-mini)
Railway                   Azure Container Apps
   |                         |
   +----------+--------------+
              |  (place clusters per day)
              v
           Agent 3
        Day Scheduler
   :8080 | FastA2A service
   GCP Cloud Run
        |            |
   MCP tool      Agent 4 (external)
   :8090          Food Recommender
 Google Routes    (not in this repo)
```

All inter-agent communication uses the **A2A (Agent-to-Agent) protocol** — a standard HTTP JSON-RPC interface. Every agent exposes three endpoints:

- `GET /health` — liveness check
- `GET /.well-known/agent.json` — describes the agent's capabilities
- `POST /` — accepts and executes tasks

---

## Agent-by-Agent Breakdown

### Agent 0 — Orchestrator (`orchestrator-agent/`)

The **entry point** of the whole system. It is a FastA2A server that:

- Accepts a natural-language trip request from the user.
- Runs an **LLM-driven tool-calling loop** (using OpenRouter, default model: `google/gemini-2.0-flash-001`).
- Calls Agent 1, Agent 2, and Agent 3 in sequence via the A2A protocol.
- Collects all results and returns a final structured itinerary as an A2A task artifact.

The orchestration flow is **LLM-driven, not hardcoded** — the LLM itself decides when and how to call each downstream agent using tool definitions.

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Required — LLM API key |
| `OPENROUTER_MODEL` | Optional override (default: `google/gemini-2.0-flash-001`) |
| `AGENT1_URL` | URL of Place Recommender (default: `http://localhost:8000`) |
| `AGENT2_URL` | URL of Place Clusterer (default: `http://localhost:8001`) |
| `AGENT3_URL` | URL of Day Scheduler (default: `http://localhost:8003`) |

---

### Agent 1 — Visiting Place Recommender (`visiting-place-recommender/`)

Responsible for **finding tourist places** in the destination city. It:

- Receives trip details (destination, preferences, budget, number of days).
- Uses a **pydantic-ai** agent backed by a configurable LLM (supports OpenAI, Azure OpenAI, OpenRouter, Gemini, Anthropic, and more).
- Uses a **Google Maps MCP toolset** to search for real places via the Google Maps API.
- Returns a structured list of candidate places with: name, location (lat/lon), estimated visit duration, cost, category, rating, and opening hours.

The system prompt instructs this agent to prioritize places that are strong candidates for the final trip schedule, considering location, user preferences, and proximity to accommodation.

Deployed on **Railway**.

---

### Agent 2 — Visiting Place Clusterer (`visiting-place-clusterer/`)

Receives the full list of recommended places and **groups them into day-clusters**. It:

- Uses a pydantic-ai agent backed by **Azure OpenAI** (`gpt-4o-mini`) via Azure AI Foundry.
- Applies a `ClusteringService` that divides places across the number of trip days, considering geography and logical grouping.
- Returns a list of day-clusters, each containing the places best visited together on the same day.

Deployed on **Azure Container Apps** with a fixed single replica (FastA2A keeps task state in-memory, so horizontal scaling is not allowed).

---

### Agent 3 — Day Scheduler (`services/agent-3/`)

The **most sophisticated agent** in the system. Produces a detailed, hour-by-hour schedule for a single day:

- Accepts a `DaySchedulingRequest` with a list of places, a time window (`day_start` / `day_end`), food budget, preferences, and allowed transport modes.
- **Sorts places by `priority_score`** and fits them within opening hours and the daily time window.
- Calls the **Agent 3 MCP service** to get real travel time and distance from the Google Maps Routes API between each location.
- Inserts **lunch** (12:00–14:00) and **dinner** (19:00–21:00) events by calling **Agent 4** (external Food Recommender).
- Returns a chronological list of events (`visit`, `travel`, `meal`), any unscheduled places, and warning messages.

Key planner behaviours:
- Places that cannot fit in the time window are moved to `unscheduled_places`.
- Travel events are inserted between every pair of consecutive locations.
- Meal travel is deferred until near the meal window — the user is not sent to the restaurant early.
- If Agent 4 is configured but returns no results, Agent 3 falls back to a synthetic meal event and emits a warning.
- If Agent 4 is not configured (`AGENT3_AGENT4_BASE_URL` unset), meal-window requests fail with `agent4_unconfigured`.

Deployed on **GCP Cloud Run** (private, IAM-protected).

---

### Agent 3 MCP — Route Tool Service (`services/agent-3-mcp/`)

A **FastAPI tool microservice** (not an LLM agent itself) that Agent 3 calls for real-world routing data. Exposes:

- `POST /v1/tools/route-estimate` — calls the **Google Maps Routes API** and returns estimated travel distance and duration. Supports `walking`, `driving`, `transit`, and `bicycling`.
- `POST /v1/tools/place-details` — placeholder (not yet implemented).

The Google Maps API key is stored in **GCP Secret Manager** in production and injected at runtime. Deployed on **GCP Cloud Run** (private); Agent 3 authenticates using Google-signed ID tokens.

---

### Agent 4 — Food Recommender (external / `food-place-recommender/`)

Suggests restaurants for meal events. This is an **external dependency** — `food-place-recommender/` contains a reference implementation but Agent 4 is **not deployed** by this repo's CI/CD pipeline. Agent 3 integrates with it via `AGENT3_AGENT4_BASE_URL`.

When a meal window is reached:
1. Agent 3 calls Agent 4 via A2A with meal context (budget, preferences, location, time).
2. Agent 4 returns a list of restaurant candidates.
3. Agent 3 inserts the best match as a `meal` event in the schedule.

---

### Legacy / Prototype Services

These earlier-iteration prototypes are preserved for reference but are **not part of the active CI/CD pipeline**:

- `orchestrator/` — early boilerplate for the orchestrator.
- `single-day-plan-scheduler/` + `single-day-plan-scheduler-mcp/` — prototype versions of the day scheduler and its MCP tool service.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Agent framework | pydantic-ai, FastA2A |
| Inter-agent protocol | A2A (Agent-to-Agent, JSON-RPC over HTTP) |
| Tool protocol | MCP (Model Context Protocol) |
| LLM providers | OpenRouter (Gemini 2.0 Flash), Azure OpenAI (gpt-4o-mini) |
| External APIs | Google Maps Places API, Google Maps Routes API |
| Language | Python 3.12+ |
| Dependency management | uv |
| Linting | ruff |
| Web framework | FastAPI / Starlette (ASGI) |
| Containerization | Docker, Docker Compose |
| GCP deployment | Cloud Run, Artifact Registry, Secret Manager, Cloud Build |
| Azure deployment | Azure Container Apps, Azure AI Foundry, Azure Container Registry |
| Railway deployment | Railway (visiting-place-recommender) |
| CI/CD | GitHub Actions (ci, deploy, live-smoke workflows) |
| Testing | pytest (mocked unit tests) |

---

## Repository Structure

```
.
├── .github/
│   ├── workflows/
│   │   ├── ci.yml           # Lint + test on every PR and push
│   │   ├── deploy.yml       # Deploy to GCP Cloud Run on merge to main
│   │   └── live-smoke.yml   # Advisory integration smoke tests (nightly)
│   └── scripts/             # Deploy helper scripts
│
├── docs/
│   ├── ci-cd.md             # Full CI/CD architecture documentation
│   └── gcp-cloud-run.md     # GCP Cloud Run operator runbook
│
├── services/                # ACTIVELY DEPLOYED services
│   ├── agent-3/             # Day Scheduler (GCP Cloud Run)
│   └── agent-3-mcp/         # Route tool service for Agent 3 (GCP Cloud Run)
│
├── orchestrator-agent/           # Agent 0 - Trip Planner Orchestrator
├── visiting-place-recommender/   # Agent 1 - Place Recommender (Railway)
├── visiting-place-clusterer/     # Agent 2 - Place Clusterer (Azure Container Apps)
├── food-place-recommender/       # Agent 4 - Food Recommender (reference impl.)
│
├── orchestrator/                 # Early boilerplate (not deployed)
├── single-day-plan-scheduler/    # Prototype scheduler (not deployed)
├── single-day-plan-scheduler-mcp/# Prototype MCP service (not deployed)
│
├── tools/cicd/                   # Python tooling for manifest-driven deployment
├── tests/cicd/                   # CI/CD infrastructure tests
└── docker-compose.yml            # Local full-stack orchestration
```

---

## Running Locally

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager used by all services)
- Docker & Docker Compose
- A Google Maps API key
- An OpenRouter API key (or any supported LLM provider key)

---

### Option A — Docker Compose (full stack)

The root `docker-compose.yml` brings up all services on a shared `travel-scheduler-network`:

| Service | Host Port |
|---|---|
| orchestrator | 8000 |
| visiting-place-recommender | 8001 |
| visiting-place-clusterer | 8002 |
| single-day-plan-scheduler | 8003 |
| food-place-recommender | 8004 |

```bash
# Copy and fill the .env file for each service
cp visiting-place-recommender/.env.example visiting-place-recommender/.env
# ... repeat for each service, adding your API keys

docker-compose up --build
```

---

### Option B — Run Agent 3 + MCP manually

```bash
# 1. Start Agent 3 MCP (route tool service) on port 8090
cd services/agent-3-mcp
uv sync --group dev
AGENT3_MCP_GOOGLE_MAPS_API_KEY=<your_key> \
  uv run python -m uvicorn agent3_mcp.main:create_app --factory --reload \
  --host 127.0.0.1 --port 8090

# 2. Start Agent 3 (day scheduler) on port 8080
cd ../agent-3
uv sync --group dev
AGENT3_PUBLIC_BASE_URL=http://127.0.0.1:8080 \
AGENT3_MCP_BASE_URL=http://127.0.0.1:8090 \
AGENT3_MCP_AUTH_MODE=none \
  uv run python -m uvicorn agent3.main:create_app --factory --reload \
  --host 127.0.0.1 --port 8080
```

---

### Option C — Run the Orchestrator

```bash
cd orchestrator-agent
pip install -r requirements.txt
cp .env.example .env   # set OPENROUTER_API_KEY inside
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

### Sending a Test Request to Agent 3

```bash
curl -X POST http://127.0.0.1:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "schedule-1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "msg-1",
        "role": "user",
        "kind": "message",
        "parts": [{
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
                "opening_hours": [{
                  "day_of_week": "monday",
                  "open_time": "09:00:00",
                  "close_time": "18:00:00"
                }]
              }
            ]
          }
        }]
      },
      "configuration": {"acceptedOutputModes": ["application/json"]}
    }
  }'
```

---

## Cloud Deployment

### GCP Cloud Run — `services/agent-3` and `services/agent-3-mcp`

These are the **only two services with automated CI/CD deployment** from this repository.

- **Project:** `cloud-computing-course-495606`, region: `europe-west1`
- Container images are built via **Cloud Build** and stored in **Artifact Registry** (`agent-services` repository).
- Secrets (e.g., Google Maps API key) are stored in **GCP Secret Manager** and injected at runtime.
- Both services are **private** (Cloud Run IAM-protected). Agent 3 authenticates to Agent 3 MCP using Google-signed ID tokens.
- Agent 3 is deployed in a **two-pass process**: first deploy, then redeploy with `AGENT3_PUBLIC_BASE_URL` set to the real Cloud Run URL so the FastA2A agent card advertises the correct endpoint.

For full setup instructions, see [docs/gcp-cloud-run.md](docs/gcp-cloud-run.md).

---

### Azure Container Apps — `visiting-place-clusterer`

- **Resource group:** `polimi-cloud`, region: `uksouth`
- LLM: **Azure AI Foundry** hosting `gpt-4o-mini` (Azure OpenAI)
- Container images are built locally and pushed to **Azure Container Registry** (ACR Tasks are blocked on Azure for Students subscriptions).
- Must run as a **single replica** — FastA2A is stateful and cannot scale horizontally.

---

### Railway — `visiting-place-recommender`

Deployed on Railway with environment variables configured via the Railway dashboard.

---

## CI/CD Pipeline

The repository uses three **GitHub Actions** workflows. See [docs/ci-cd.md](docs/ci-cd.md) for the full architecture.

### `ci.yml` — Continuous Integration

Triggered on every push and pull request affecting service code. For each changed service it runs:

1. `uv sync --group dev` — install dependencies
2. `uv run ruff check .` — lint
3. `uv run pytest` — run tests

### `deploy.yml` — Continuous Deployment

Triggered on every merge to `main`. For each service with `ownership: repo-deployed` and `backend: gcp-cloud-run` in its `service.deploy.yaml` manifest:

1. Builds the container image via Cloud Build.
2. Pushes to Artifact Registry.
3. Deploys to Cloud Run in dependency order (`agent-3-mcp` always before `agent-3`).
4. Runs post-deploy smoke checks: `/health`, agent card, and a real FastA2A scheduling task.

GitHub Actions authenticates to GCP using **Workload Identity Federation (WIF)** — no long-lived JSON key required.

### `live-smoke.yml` — Advisory Smoke Tests

Run nightly and on manual dispatch. Tests real external integrations (Google Routes API, external Agent 4). These are **advisory only** and do not block deployment.

---

## Environment Variables Reference

### Agent 3 (`services/agent-3`)

| Variable | Required | Description |
|---|---|---|
| `AGENT3_PUBLIC_BASE_URL` | Yes (deploy) | Publicly advertised URL of this service |
| `AGENT3_MCP_BASE_URL` | Yes | Base URL for the Agent 3 MCP route service |
| `AGENT3_MCP_AUTH_MODE` | No | `none` locally; `gcp_id_token` on Cloud Run |
| `AGENT3_AGENT4_BASE_URL` | Optional | External Agent 4 URL; required for meal-window requests |
| `AGENT3_AGENT4_TIMEOUT_SECONDS` | No | HTTP timeout for Agent 4 calls |
| `AGENT3_FALLBACK_TRAVEL_MINUTES` | No | Travel time fallback used when MCP route fails |
| `AGENT3_LOG_LEVEL` | No | Application log level |

### Agent 3 MCP (`services/agent-3-mcp`)

| Variable | Required | Description |
|---|---|---|
| `AGENT3_MCP_GOOGLE_MAPS_API_KEY` | Yes | Google Maps Platform API key |
| `AGENT3_MCP_GOOGLE_ROUTES_TIMEOUT_SECONDS` | No | HTTP timeout for Routes API requests |
| `AGENT3_MCP_LOG_LEVEL` | No | Application log level |

### Orchestrator (`orchestrator-agent`)

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | LLM API key via OpenRouter |
| `OPENROUTER_MODEL` | No | Model override (default: `google/gemini-2.0-flash-001`) |
| `AGENT1_URL` | No | Place Recommender URL |
| `AGENT2_URL` | No | Place Clusterer URL |
| `AGENT3_URL` | No | Day Scheduler URL |

> Each service's `.env.example` file contains the full list of supported variables.

---

## Testing

Each service has its own test suite. Run from inside the service directory:

```bash
# Lint
uv run ruff check .

# Run all tests
uv run pytest

# Agent 3 - unit tests only
uv run pytest -m unit

# Agent 3 - exclude integration tests
uv run pytest -m "not integration"
```

- **Agent 3** unit tests are fully mocked — downstream agents and the LLM are mocked, so tests run without external dependencies.
- **Orchestrator** tests live in `orchestrator-agent/tests/test_orchestrator_mocked.py`.
- **Live integration smoke tests** are handled separately by `live-smoke.yml`.

---

## Contributors

| Name | GitHub |
|---|---|
| Alireza Jahandoost | [@alireza-jahandoost](https://github.com/alireza-jahandoost) |
| Javad Zandiyeh | [@JavadZandiyeh](https://github.com/JavadZandiyeh) |
| Aman Zargari | [@amanzargari](https://github.com/amanzargari) |
| Moein Taherine | [@MoeinDSP](https://github.com/MoeinDSP) |

---

Licensed under the **MIT License**.

> This project was created as a Microsoft-sponsored demo extension for a cloud computing course at Politecnico di Milano. It demonstrates multi-agent AI system design, the A2A inter-agent protocol, MCP tool integration, multi-cloud deployment (GCP, Azure, Railway), and CI/CD automation with GitHub Actions.
