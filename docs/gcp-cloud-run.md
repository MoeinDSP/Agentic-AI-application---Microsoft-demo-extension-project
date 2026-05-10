# GCP Cloud Run Adapter

This document covers the GCP-specific adapter used by the repo-wide CI/CD
framework documented in [ci-cd.md](./ci-cd.md).

The current repo-owned Cloud Run services are:

- `agent-3`
- `agent-3-mcp`

Agent 4 remains external and must be provided through
`AGENT3_AGENT4_BASE_URL`.

## Required GCP Setup

1. Enable APIs:
   - Cloud Run Admin API
   - Artifact Registry API
   - Secret Manager API
   - Cloud Build API
   - Maps Routes API
2. Create an Artifact Registry repository for container images.
3. Create a Google Maps API key with Routes API access enabled.
4. Store that API key in Secret Manager.

Example:

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com cloudbuild.googleapis.com routes.googleapis.com

gcloud secrets create agent3-mcp-google-maps-api-key --replication-policy="automatic"
printf '%s' 'YOUR_GOOGLE_MAPS_API_KEY' | \
  gcloud secrets versions add agent3-mcp-google-maps-api-key --data-file=-
```

## Artifact Registry Defaults

Chosen defaults for the current setup:

```bash
export PROJECT_ID="cloud-computing-course-495606"
export REGION="europe-west1"
export REPOSITORY="agent-services"
```

You can override these through GitHub repository or environment variables:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY`

## Validation

## GitHub-Side Configuration

Repository or environment variables:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY`
- `GCP_SERVICE_ACCOUNT_EMAIL`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `AGENT3_AGENT4_BASE_URL`

Optional secret fallback:

- `GCP_CREDENTIALS_JSON`

Preferred auth is Workload Identity Federation via
`google-github-actions/auth`. The credentials JSON fallback exists only to
unblock the first rollout before WIF is configured.

## Deployment Flow

The deploy workflow:

1. discovers changed repo-deployed services
2. filters to `backend: gcp-cloud-run`
3. deploys them in dependency order
4. runs post-deploy smoke checks

For the current services:

- `agent-3-mcp` is deployed before `agent-3` when both change
- `agent-3` receives the resolved Cloud Run URL of `agent-3-mcp`
- `agent-3-mcp` receives the Google Maps key from Secret Manager
- `agent-3` receives `AGENT3_AGENT4_BASE_URL` from GitHub configuration

## Post-Deploy Validation

After a deployment, the adapter runs service-type smoke checks.

Current checks:

```bash
curl https://AGENT3_URL/health
curl https://AGENT3_MCP_URL/health
```

- Agent 3 can reach Agent 3 MCP
- Agent 3 MCP can reach Google Routes
- Agent 3 can reach the configured external Agent 4 endpoint
- Agent 3 FastA2A scheduling tasks complete successfully

## Rollback / Redeploy

Redeploy by:

1. re-running the `deploy.yml` workflow for a known good commit, or
2. reverting the merge on `main`, which will trigger a new deploy

Cloud Run revision history remains available in GCP for operational rollback.

## Remaining Work

- `place-details` remains placeholder.
- Remote Agent 4 is still outside your GCP project.
- Runtime auth hardening remains a separate follow-up from CI/CD itself.
