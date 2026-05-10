# CI/CD Architecture

This repository uses a repo-wide CI/CD framework that is intentionally broader
than the current deployment scope.

The current repo-owned deployable services are:

- `services/agent-3`
- `services/agent-3-mcp`

Other services can be merged into the monorepo later without being deployed
automatically, as long as they do not opt into the deployment contract.

## Deployment Model

- **CI orchestrator:** GitHub Actions
- **CD orchestrator:** GitHub Actions
- **Current deploy backend:** GCP Cloud Run
- **Current image registry:** Artifact Registry
- **Current secret store for runtime secrets:** Google Secret Manager
- **Current release strategy:** auto-deploy on merge to `main`
- **Current environment model:** production only

GitHub Actions is the repo-level control plane so future services can add
non-GCP deploy adapters without redesigning the monorepo workflow model.

## Service Deployment Contract

Each deployable service defines `service.deploy.yaml` under its service root.

Required top-level fields:

- `service_name`
- `ownership`
- `deploy_enabled`
- `validate_enabled`
- `backend`
- `build_context`
- `dockerfile`
- `runtime_port`
- `healthcheck`
- `smoke_check_type`
- `depends_on`
- `ci`
- `deploy`

Current ownership modes:

- `repo-deployed`: this repo’s CD pipeline owns deployment
- `external`: the service is used by this repo but deployed elsewhere
- `undeployed`: the service exists in source control but is not part of CD

## Workflows

### `ci.yml`

Runs on pushes and pull requests that affect service code or CI/CD
infrastructure.

Behavior:

1. Discover changed services from manifests and changed files
2. Run repo-level CI/CD tests
3. Run service-level `uv`/lint/test validation for affected services

### `live-smoke.yml`

Runs on manual dispatch and nightly schedule.

Behavior:

- uses the manifest-declared live smoke command per service
- runs the real Google Routes smoke test for `agent-3-mcp`
- runs the external Agent 4 smoke test for `agent-3` when
  `AGENT3_AGENT4_BASE_URL` is configured

This workflow is intentionally separate from PR gating.

### `deploy.yml`

Runs on push to `main`.

Behavior:

1. Discover changed repo-deployed services
2. Filter to `backend: gcp-cloud-run`
3. Deploy them in dependency order
4. Run post-deploy smoke checks

## GitHub Configuration

Repository or environment variables:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY`
- `GCP_SERVICE_ACCOUNT_EMAIL`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `AGENT3_AGENT4_BASE_URL`

Optional GitHub secret fallback:

- `GCP_CREDENTIALS_JSON`

The preferred authentication path is Workload Identity Federation. The JSON key
fallback exists only so the workflow can still run before WIF is configured.

## GCP Adapter

The GCP Cloud Run adapter:

- builds container images with `gcloud builds submit`
- pushes them to Artifact Registry
- deploys them with `gcloud run deploy`
- resolves dependency service URLs from Cloud Run
- injects runtime secrets from Secret Manager
- performs post-deploy smoke checks by service type

Current dependency rule:

- `agent-3` depends on `agent-3-mcp`

If both change in one merge, `agent-3-mcp` deploys first.

## Runtime Secret Ownership

- `AGENT3_MCP_GOOGLE_MAPS_API_KEY` is a runtime secret owned by GCP Secret Manager
- `AGENT3_AGENT4_BASE_URL` is a deployment-time config value owned by GitHub variables

Agent 4 is external and env-driven. It is not deployed from this repository.
