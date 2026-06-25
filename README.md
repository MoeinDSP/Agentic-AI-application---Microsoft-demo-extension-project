# Agentic-AI-application---Microsoft-demo-extension-project

## Services

- `services/agent-3`: Agent 3 day scheduling service.
- `services/agent-3-mcp`: independent route/place tool service for Agent 3.

## CI/CD

- Repo-wide CI/CD architecture: [docs/ci-cd.md](docs/ci-cd.md)
- GCP Cloud Run adapter details: [docs/gcp-cloud-run.md](docs/gcp-cloud-run.md)



User Request (natural language)
        │
        ▼
┌─────────────────────────────────────┐
│   Agent 0 — Orchestrator            │
│   (FastA2A, LLM tool-calling loop)  │
│   Port :8080 | OpenRouter/Gemini    │
└───────┬─────────────────────────────┘
        │ calls via A2A protocol
    ┌───┴──────────────────────────┐
    │                              │
    ▼                              ▼
Agent 1                        Agent 2
Place Recommender              Place Clusterer
:8000 | pydantic-ai            :8001 | pydantic-ai
Google Maps MCP                Azure OpenAI (gpt-4o-mini)
Railway deployment             Azure Container Apps
    │                              │
    └───────────────┬──────────────┘
                    │ clusters of places
                    ▼
               Agent 3
           Day Scheduler
      :8080 | FastA2A service
      GCP Cloud Run deployment
           │         │
     MCP tool    Agent 4 (external)
     :8090        Food Recommender
  Google Routes    (not in this repo)








  .
├── .github/
│   ├── workflows/
│   │   ├── ci.yml           # Lint, test on PRs and pushes
│   │   ├── deploy.yml       # Deploy to GCP Cloud Run on merge to main
│   │   └── live-smoke.yml   # Advisory smoke tests (nightly + manual)
│   └── scripts/             # Deploy helper scripts
│
├── docs/
│   ├── ci-cd.md             # Full CI/CD architecture documentation
│   └── gcp-cloud-run.md     # GCP Cloud Run operator runbook
│
├── services/                # ACTIVELY DEPLOYED services
│   ├── agent-3/             # Day Scheduler (GCP Cloud Run)
│   └── agent-3-mcp/         # Route/Tool service for Agent 3 (GCP Cloud Run)
│
├── orchestrator-agent/      # Agent 0 — Trip Planner Orchestrator
├── visiting-place-recommender/   # Agent 1 — Place Recommender (Railway)
├── visiting-place-clusterer/     # Agent 2 — Place Clusterer (Azure Container Apps)
├── food-place-recommender/       # Agent 4 — Food Recommender (reference impl.)
│
├── orchestrator/            # Early boilerplate (not deployed)
├── single-day-plan-scheduler/         # Prototype scheduler (not deployed)
├── single-day-plan-scheduler-mcp/     # Prototype MCP service (not deployed)
│
├── tools/
│   └── cicd/                # Python tooling for manifest-driven deployment
│
├── tests/
│   └── cicd/                # CI/CD infrastructure tests
│
└── docker-compose.yml       # Local full-stack orchestration
