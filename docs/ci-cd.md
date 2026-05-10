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

Live smoke is advisory, not release-blocking:

- failures in `live-smoke.yml` should be treated as integration diagnostics
- they do not block `deploy.yml`
- use them to validate external dependencies and richer scenarios after deploy,
  not to decide whether the repo-owned services basically deployed correctly

### `deploy.yml`

Runs on push to `main`.

Behavior:

1. Discover changed repo-deployed services
2. Filter to `backend: gcp-cloud-run`
3. Deploy them in dependency order
4. Run post-deploy smoke checks

This flow has now been exercised successfully against the current private Cloud
Run deployment path, including authenticated post-deploy smoke.

Changes to shared deployment infrastructure under `.github/scripts/`,
`.github/workflows/`, `tools/cicd/`, or `tests/cicd/` intentionally trigger
deployment of all repo-deployed services on the affected backend. This ensures
deploy-system changes are exercised against real services instead of requiring
fake no-op edits under `services/`.

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

### Repository or environment secrets

- `AGENT3_MCP_GOOGLE_MAPS_API_KEY`

### Optional repository or environment variables

- `AGENT3_AGENT4_BASE_URL`

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
  - external runtime integration config for `agent-3`
  - owned by GitHub variables
  - optional for deployment
  - required only for requests that actually cross lunch or dinner windows

## Post-Deploy Guarantees

The current deploy smoke checks are now verified in production and guarantee:

- `agent-3-mcp` responds on `/health`
- `agent-3-mcp` can answer a real `/v1/tools/route-estimate` request
- `agent-3` responds on `/health`
- `agent-3` exposes an agent card
- `agent-3` completes a FastA2A scheduling task
- the Agent 3 agent-card URL matches the deployed Cloud Run service URL

Authenticated smoke uses Google-signed ID tokens minted through the deployer
service account via impersonation. This is part of the proven deployment path
for private Cloud Run services.

Current limitation:

- deploy smoke intentionally does **not** verify external Agent 4 reachability
- deploy smoke intentionally avoids meal windows so deployment success depends on
  Agent 3 itself, not on a third-party external meal recommender
- route fallback warnings are not treated as deploy failures; they are runtime
  behavior signals to investigate separately

Operational note:

- Agent 3 is intended to be redeployed automatically with
  `AGENT3_PUBLIC_BASE_URL` set to the resolved Cloud Run URL
- that automation exists and is the intended steady state
- after this rollout, the agent-card URL should still be explicitly verified
  post-deploy because one production rollout required a manual correction
- meal-window validation should be run separately as a recommended post-deploy
  check once `AGENT3_AGENT4_BASE_URL` is configured

## GCP Adapter

The GCP Cloud Run adapter:

- builds container images with `gcloud builds submit`
- pushes them to Artifact Registry
- deploys them with `gcloud run deploy`
- submits Cloud Build jobs asynchronously and polls them to completion
- resolves dependency service URLs from Cloud Run
- injects runtime secrets from Secret Manager
- performs post-deploy smoke checks by service type

For Agent 3 specifically, deployment is a two-step update:

1. deploy the service
2. read the deployed Cloud Run URL
3. redeploy Agent 3 with `AGENT3_PUBLIC_BASE_URL` set to that URL

That second step is required so the FastA2A agent card advertises the real
production URL instead of a local default.

## Current Verified State

The following outcomes have been verified against the current production
deployment:

- `agent-3-mcp /health` succeeded on the deployed private Cloud Run service
- `agent-3 /health` succeeded on the deployed private Cloud Run service
- Agent 3 agent-card URL now matches the deployed Cloud Run URL
- an authenticated no-meal FastA2A task completed successfully
- an authenticated meal-window FastA2A task completed successfully after
  `AGENT3_AGENT4_BASE_URL` was configured
- the meal event in that production task was non-synthetic and came from the
  external Agent 4 service
- route fallback warnings were still observed on some hops during the meal-window
  test

These observations are operational evidence from the current rollout, not a
guarantee that every future request will be warning-free.

## Current Security State

The current target security model is:

- GitHub Actions authenticates to GCP with WIF for deployment
- `agent-3` and `agent-3-mcp` are deployed as authenticated Cloud Run services
- external callers authenticate to Agent 3 through Cloud Run IAM
- Agent 3 authenticates to MCP with a Google-signed ID token

App-level custom auth middleware is intentionally not part of this design.
