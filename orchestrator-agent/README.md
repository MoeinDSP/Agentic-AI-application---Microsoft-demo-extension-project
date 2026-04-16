# Agent 0 — Trip Planner Orchestrator

Main entry point of the trip planner multi-agent system.

## Architecture

Receives a user trip request and coordinates:
1. **Agent 1** (Place Recommender) — via A2A at `:8000`
2. **Agent 2** (Place Clustering) — via REST at `:8001`
3. **Agent 3** (Daily Scheduler) — built into orchestrator
4. **Agent 4** (Food Recommender) — via A2A at `:8004`

## Quick Start

```bash
cp .env.example .env
# Fill in your Google Cloud credentials

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Or with Docker
docker-compose up --build
```

## Test

```bash
pytest -m unit        # unit tests only
pytest -m integration # requires all agents running
```
