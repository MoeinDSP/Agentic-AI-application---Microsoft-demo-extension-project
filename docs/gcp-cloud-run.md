# GCP Cloud Run Deployment

This repository deploys two services to Cloud Run for the real Agent 3 path:

- `agent-3`
- `agent-3-mcp`

The external food recommender remains the remote FastA2A Agent 4 endpoint at:

- `http://65.21.48.155:8004`

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

## Build Images

Example variables:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="europe-west1"
export REPOSITORY="agent-services"
export TAG="$(git rev-parse --short HEAD)"
```

Build and push:

```bash
gcloud builds submit services/agent-3 \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/agent-3:${TAG}"

gcloud builds submit services/agent-3-mcp \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/agent-3-mcp:${TAG}"
```

## Deploy Agent 3 MCP

```bash
gcloud run deploy agent-3-mcp \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/agent-3-mcp:${TAG}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --set-env-vars "AGENT3_MCP_ENVIRONMENT=production,AGENT3_MCP_GOOGLE_ROUTES_TIMEOUT_SECONDS=5.0" \
  --set-secrets "AGENT3_MCP_GOOGLE_MAPS_API_KEY=agent3-mcp-google-maps-api-key:latest"
```

Save the resulting Cloud Run URL and use it as `AGENT3_MCP_BASE_URL` for Agent 3.

## Deploy Agent 3

Replace `AGENT3_MCP_BASE_URL` with the Agent 3 MCP Cloud Run URL.

```bash
gcloud run deploy agent-3 \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/agent-3:${TAG}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --set-env-vars "AGENT3_ENVIRONMENT=production,AGENT3_MCP_BASE_URL=https://AGENT3_MCP_URL,AGENT3_AGENT4_BASE_URL=http://65.21.48.155:8004,AGENT3_AGENT4_INVOCATION_MODE=a2a,AGENT3_AGENT4_POLL_INTERVAL_SECONDS=1.0,AGENT3_AGENT4_MAX_WAIT_SECONDS=15.0,AGENT3_FALLBACK_TRAVEL_MINUTES=10"
```

## Validation

After deployment:

```bash
curl https://AGENT3_URL/health
curl https://AGENT3_MCP_URL/health
```

Then send a sample `POST /v1/plan` request to Agent 3 and verify:

- Agent 3 can reach Agent 3 MCP
- Agent 3 MCP can reach Google Routes
- Agent 3 can reach the external Agent 4 endpoint
- the response still uses the existing public scheduling contract

## Remaining Work

- `place-details` remains placeholder.
- Remote Agent 4 is still outside your GCP project.
- This pass does not add CI/CD or Terraform.
