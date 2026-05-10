# GCP Cloud Run Operator Runbook

This document covers the current GCP deployment backend used by the repo-wide
CI/CD framework in [ci-cd.md](./ci-cd.md).

The current repo-owned Cloud Run services are:

- `agent-3`
- `agent-3-mcp`

Agent 4 remains external and must be provided through
`AGENT3_AGENT4_BASE_URL`.

## Chosen Defaults

- project: `cloud-computing-course-495606`
- region: `europe-west1`
- Artifact Registry repository: `agent-services`
- Secret Manager secret for Google Maps key:
  `agent3-mcp-google-maps-api-key`

## One-Time GCP Bootstrap

### 1. Enable required APIs

- Cloud Run Admin API
- Artifact Registry API
- Secret Manager API
- Cloud Build API
- IAM Credentials API
- Security Token Service API
- Routes API

Example:

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  routes.googleapis.com
```

### 2. Create Artifact Registry

```bash
gcloud artifacts repositories create agent-services \
  --repository-format=docker \
  --location=europe-west1 \
  --description="Container images for agent services"
```

### 3. Create the runtime Google Maps secret

```bash
gcloud secrets create agent3-mcp-google-maps-api-key \
  --replication-policy="automatic"

printf '%s' 'YOUR_GOOGLE_MAPS_API_KEY' | \
  gcloud secrets versions add agent3-mcp-google-maps-api-key --data-file=-
```

### 4. Create GitHub deploy auth with WIF

The deploy workflow expects GitHub Actions to authenticate through Workload
Identity Federation.

Create:

1. a deployer service account
2. a workload identity pool
3. a GitHub OIDC provider
4. a `roles/iam.workloadIdentityUser` binding from the GitHub repo to the
   deployer service account

The resulting provider resource string is stored in the GitHub variable
`GCP_WORKLOAD_IDENTITY_PROVIDER`.

## GitHub Configuration

### Variables

- `GCP_PROJECT_ID=cloud-computing-course-495606`
- `GCP_REGION=europe-west1`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY=agent-services`
- `GCP_SERVICE_ACCOUNT_EMAIL=<github deployer service account email>`
- `GCP_WORKLOAD_IDENTITY_PROVIDER=<projects/.../workloadIdentityPools/.../providers/...>`
- `GCP_AGENT3_RUNTIME_SERVICE_ACCOUNT_EMAIL=<agent-3 runtime service account email>`
- `GCP_AGENT3_MCP_RUNTIME_SERVICE_ACCOUNT_EMAIL=<agent-3-mcp runtime service account email>`
- `AGENT3_AGENT4_BASE_URL=<external Agent 4 base URL>`

### Secrets

- `AGENT3_MCP_GOOGLE_MAPS_API_KEY=<real Google Maps key>`

This GitHub secret is used only by `live-smoke.yml`. Production runtime reads
the Google Maps key from Secret Manager, not from GitHub.

### Optional fallback secret

- `GCP_CREDENTIALS_JSON`

Use this only if WIF is not available yet. The intended long-term deploy auth is
WIF, not a JSON key.

### Recommended runtime service account naming

- `agent-3-runtime@cloud-computing-course-495606.iam.gserviceaccount.com`
- `agent-3-mcp-runtime@cloud-computing-course-495606.iam.gserviceaccount.com`

## Current Deployment Flow

On merge to `main`, `deploy.yml`:

1. discovers changed repo-deployed services
2. filters to `backend: gcp-cloud-run`
3. deploys them in dependency order
4. runs post-deploy smoke checks

For the current services:

1. `agent-3-mcp` deploys first
2. `agent-3` deploys second
3. `agent-3` receives the resolved Cloud Run URL of `agent-3-mcp`
4. `agent-3-mcp` receives the Google Maps key from Secret Manager
5. `agent-3` receives `AGENT3_AGENT4_BASE_URL` from GitHub variables
6. both services are deployed with their configured runtime service accounts
7. `agent-3` is redeployed with `AGENT3_PUBLIC_BASE_URL` set to its real Cloud
   Run URL so its FastA2A agent card advertises the correct endpoint
8. `agent-3` is deployed with `AGENT3_MCP_AUTH_MODE=gcp_id_token`

## Post-Deploy Validation

The adapter runs service-type smoke checks after deployment.

### `agent-3-mcp`

- `GET /health`
- `POST /v1/tools/route-estimate`

This validates:

- the Cloud Run service is alive
- the route endpoint works
- Google Routes integration works with the deployed runtime secret

### `agent-3`

- `GET /health`
- `GET /.well-known/agent-card.json`
- `POST /` FastA2A `message/send`
- `POST /` FastA2A `tasks/get`

This validates:

- the Cloud Run service is alive
- the FastA2A surface is reachable
- a scheduling task completes
- the agent-card URL matches the deployed Cloud Run URL

Important limitation:

- the Agent 3 deploy smoke intentionally avoids meal windows
- it validates Agent 3 itself and its advertised URL
- it does **not** validate external Agent 4 reachability by design

External Agent 4 validation belongs in `live-smoke.yml`, not in the deploy gate.

## Runtime Auth Model

### External orchestrator -> Agent 3

- private Cloud Run service
- Workload Identity Federation from the external orchestrator into Google Cloud
- Google-signed ID token presented to Cloud Run
- mapped identity granted `roles/run.invoker` on Agent 3

### Agent 3 -> Agent 3 MCP

- private Cloud Run service for MCP
- IAM grants `roles/run.invoker` to Agent 3's runtime service account
- Agent 3 sends a Google-signed ID token with audience = MCP service URL

Important rule:

- IAM and ID tokens are complementary, not alternatives
- IAM decides who is allowed
- the ID token is what the caller presents on each request

### GitHub deploy auth

- GitHub Actions uses WIF into the deployer service account
- the deployer service account needs deployment permissions plus
  `roles/iam.serviceAccountUser` on:
  - `GCP_AGENT3_RUNTIME_SERVICE_ACCOUNT_EMAIL`
  - `GCP_AGENT3_MCP_RUNTIME_SERVICE_ACCOUNT_EMAIL`

## Rollback / Redeploy

Rollback options:

1. rerun `deploy.yml` for a known good commit
2. revert the merge on `main` and let CD deploy the reverted state
3. use Cloud Run revision history for operational rollback if needed

## Known Behavioral Boundaries

- Agent 3 deploy smoke does not exercise Agent 4
- Agent 3 can still fall back to synthetic meals at runtime if Agent 4 is down
- Agent 3 MCP `place-details` remains placeholder
- meal travel is deferred until near the meal window instead of sending the user
  to the restaurant too early
- route estimates carry `departure_time` from Agent 3 through MCP into Google
  Routes
- local development keeps `AGENT3_MCP_AUTH_MODE=none`; Cloud Run deployment sets
  `AGENT3_MCP_AUTH_MODE=gcp_id_token`
