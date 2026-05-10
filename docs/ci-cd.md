# CI/CD Architecture

This repository uses a repo-wide CI/CD framework that is broader than the
current deployment scope, but only two services are deployed from this repo
today:

- `services/agent-3`
- `services/agent-3-mcp`

Other services can live in the monorepo without being deployed automatically as
long as they do not opt into the deployment contract.

## Deployment Model

- **CI orchestrator:** GitHub Actions
- **CD orchestrator:** GitHub Actions
- **Current deploy backend:** GCP Cloud Run
- **Current image registry:** Artifact Registry
- **Current runtime secret store:** Google Secret Manager
- **Release trigger:** merge to `main`
- **Environment model:** production only

GitHub Actions remains the repo-level control plane so future services can add
non-GCP deploy adapters without redesigning the monorepo workflow model.

## Service Deployment Contract

Each deployable service defines `service.deploy.yaml` under its service root.

Current ownership modes:

- `repo-deployed`: this repo's CD pipeline owns deployment
- `external`: this repo integrates with the service, but another system deploys it
- `undeployed`: the service exists in source control but is not part of CD

Current ownership in this repo:

- `agent-3`: `repo-deployed`
- `agent-3-mcp`: `repo-deployed`
- Agent 4: external dependency only, configured through `AGENT3_AGENT4_BASE_URL`

## Workflows

### `ci.yml`

Runs on pushes and pull requests that affect service code or CI/CD
infrastructure.

Behavior:

1. Discover changed services from manifests and changed files
2. Run repo-level CI/CD tests
3. Run service-level `uv sync`, `ruff`, and `pytest` for affected services

### `live-smoke.yml`

Runs on manual dispatch and nightly schedule.

Behavior:

- uses the manifest-declared live smoke command per service
- runs the real Google Routes smoke test for `agent-3-mcp`
- runs the external Agent 4 smoke test for `agent-3` when
  `AGENT3_AGENT4_BASE_URL` is configured

This workflow is intentionally separate from PR gating and separate from deploy
success criteria.

### `deploy.yml`

Runs on push to `main`.

Behavior:

1. Discover changed repo-deployed services
2. Filter to `backend: gcp-cloud-run`
3. Deploy them in dependency order
4. Run post-deploy smoke checks

Current dependency rule:

- `agent-3` depends on `agent-3-mcp`

If both change in one merge, `agent-3-mcp` deploys first.

## GitHub Configuration

### Repository or environment variables

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY`
- `GCP_SERVICE_ACCOUNT_EMAIL`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_AGENT3_RUNTIME_SERVICE_ACCOUNT_EMAIL`
- `GCP_AGENT3_MCP_RUNTIME_SERVICE_ACCOUNT_EMAIL`
- `AGENT3_AGENT4_BASE_URL`

### Repository or environment secrets

- `AGENT3_MCP_GOOGLE_MAPS_API_KEY`

### Optional fallback secret

- `GCP_CREDENTIALS_JSON`

Use Workload Identity Federation as the preferred deploy auth path. The JSON key
fallback exists only to unblock deployment before WIF is configured.

## Runtime Secret and Config Ownership

- `AGENT3_MCP_GOOGLE_MAPS_API_KEY`
  - runtime secret for `agent-3-mcp`
  - owned by Google Secret Manager in production
  - also stored as a GitHub Actions secret only for `live-smoke.yml`
- `AGENT3_AGENT4_BASE_URL`
  - deployment-time config for `agent-3`
  - owned by GitHub variables
  - required because Agent 4 is external and is not deployed from this repo

## Post-Deploy Guarantees

The current deploy smoke checks guarantee:

- `agent-3-mcp` responds on `/health`
- `agent-3-mcp` can answer a real `/v1/tools/route-estimate` request
- `agent-3` responds on `/health`
- `agent-3` exposes an agent card
- `agent-3` completes a FastA2A scheduling task
- the Agent 3 agent-card URL matches the deployed Cloud Run service URL

Current limitation:

- deploy smoke intentionally does **not** verify external Agent 4 reachability
- deploy smoke intentionally avoids meal windows so deployment success depends on
  Agent 3 itself, not on a third-party external meal recommender

## GCP Adapter

The GCP Cloud Run adapter:

- builds container images with `gcloud builds submit`
- pushes them to Artifact Registry
- deploys them with `gcloud run deploy`
- resolves dependency service URLs from Cloud Run
- injects runtime secrets from Secret Manager
- performs post-deploy smoke checks by service type

For Agent 3 specifically, deployment is a two-step update:

1. deploy the service
2. read the deployed Cloud Run URL
3. redeploy Agent 3 with `AGENT3_PUBLIC_BASE_URL` set to that URL

That second step is required so the FastA2A agent card advertises the real
production URL instead of a local default.

## Current Security State

The current target security model is:

- GitHub Actions authenticates to GCP with WIF for deployment
- `agent-3` and `agent-3-mcp` are deployed as authenticated Cloud Run services
- external callers authenticate to Agent 3 through Cloud Run IAM
- Agent 3 authenticates to MCP with a Google-signed ID token

App-level custom auth middleware is intentionally not part of this design.
