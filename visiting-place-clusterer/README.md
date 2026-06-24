# Deploying the Visiting Place Clusterer to Azure

The clusterer is a containerized **A2A agent** (pydantic-ai → FastA2A). Azure AI
Foundry hosts the **LLM model**; the **container** runs on **Azure Container Apps
(ACA)** — the equivalent of where the recommender ran on Railway.

```text
Dockerfile / image  ──►  Azure Container Apps   (the agent, public HTTPS URL)
                                  │  calls
                                  ▼
        gpt-4o-mini deployment ◄──  Azure AI Foundry project (Azure OpenAI)
```

## Live resources (subscription: *Azure for Students*)

| Thing | Value |
| --- | --- |
| Resource group / region | `polimi-cloud` / `uksouth` |
| Foundry (AIServices) account | `visiting-place-clusterer` |
| Azure OpenAI endpoint | `https://visiting-place-clusterer.openai.azure.com` |
| Model deployment | `gpt-4o-mini` (gpt-4o-mini · 2024-07-18 · GlobalStandard) |
| Container registry | `ca72f0c87afdacr.azurecr.io` |
| Container Apps environment | `polimi-cloud-env` |
| Container App | `visiting-place-clusterer` |
| **Public URL** | `https://visiting-place-clusterer.icypond-686fddf2.uksouth.azurecontainerapps.io` |
| Health | `GET /health` → `200 ok` |
| A2A agent card | `GET /.well-known/agent-card.json` |
| A2A JSON-RPC | `POST /` (`message/send`, `tasks/get`) |

The agent runs with `APP_MODE=a2a` and `PROVIDER=azure`. The API key is stored as
an ACA **secret** (`azure-openai-key`) and referenced via
`AZURE_OPENAI_API_KEY=secretref:azure-openai-key`.

> **Single replica (required):** FastA2A keeps task state in memory, so the app is
> pinned to `--min-replicas 1 --max-replicas 1`. Do not scale out.

## What got created (and why)

The deployment produced these resources, all in resource group `polimi-cloud`
(UK South) under the *Azure for Students* subscription:

![Azure resources for the clusterer](images/azure-resources.png)

> Save the portal screenshot to `visiting-place-clusterer/images/azure-resources.png`
> for the image above to render.

| Name | Type | Role |
| --- | --- | --- |
| `visiting-place-clusterer` | **Container App** | The running agent itself — pulls the image, runs `uvicorn`, and serves the public HTTPS URL (`/health`, the A2A card, the JSON-RPC endpoint). This is what the orchestrator calls. Always-on at 1 replica. |
| `polimi-cloud-env` | **Container Apps Environment** | The hosting boundary (shared network + logging + domain) that Container Apps live inside. Defines the `…icypond-686fddf2.uksouth.azurecontainerapps.io` domain. Your other agents could share this same environment. |
| `visiting-place-clusterer` | **Foundry** (AIServices account) | The Azure AI Foundry / Azure OpenAI **account**. Hosts the `gpt-4o-mini` model deployment and exposes the inference endpoint (`…openai.azure.com`) that the container calls on every request. |
| `visiting-place-clusterer (…/…)` | **Foundry project** | A workspace *inside* the Foundry account that organizes models, agents, and connections. It exposes the project endpoint (`…services.ai.azure.com/api/projects/…`) used by the Foundry Agent SDK. Our container hits the account's Azure OpenAI endpoint directly, so the project isn't on the request path — it's where the model was deployed from. |
| `ca72f0c87afdacr` | **Container registry (ACR)** | Private Docker registry holding the `visiting-place-clusterer:v1` image; the Container App pulls from here. Auto-named by the first `containerapp up` attempt. |
| `workspace-polimicloud9wDu` | **Log Analytics workspace** | Log/metrics store auto-created with the environment. Backs `az containerapp logs show` and KQL queries. |
| `Azure for Students` | **Subscription** | Billing + access boundary that owns everything above; your student credit is drawn from here. |

### What costs credit

| Resource | Cost model |
| --- | --- |
| Container App | per vCPU-second + memory while running — always-on at 1 replica is a small continuous drain |
| Log Analytics | per GB of logs ingested/retained |
| Container Registry | ~Basic SKU, a few cents/day standing charge |
| Foundry model | per token on `gpt-4o-mini` (very cheap); nothing when idle |
| Environment / Foundry project / Subscription | no direct charge themselves |

To minimize burn while idle, set `--min-replicas 0` (scale to zero, ~seconds cold
start). To tear the whole thing down: `az group delete -n polimi-cloud`.

## One-time setup

```bash
az login
az extension add -n containerapp --upgrade
az provider register -n Microsoft.App
az provider register -n Microsoft.OperationalInsights
az provider register -n Microsoft.ContainerRegistry

# Model deployment in the Foundry project
az cognitiveservices account deployment create \
  -n visiting-place-clusterer -g polimi-cloud \
  --deployment-name gpt-4o-mini \
  --model-name gpt-4o-mini --model-version 2024-07-18 --model-format OpenAI \
  --sku-name GlobalStandard --sku-capacity 10
```

## Build & push the image

> ⚠️ **ACR Tasks (`az acr build`) is blocked on Azure for Students**
> (`TasksOperationsNotAllowed`). Build **locally** and push instead. ACA only runs
> **linux/amd64**, so cross-build from Apple Silicon.

The build context must contain `Dockerfile`, `pyproject.toml`, `uv.lock`,
`README.md`, and `app/` — and **not** `.env`/`.venv`. The repo path contains
spaces, which trips some tooling, so stage a clean context:

```bash
SRC="$(pwd)"   # the visiting-place-clusterer directory
rm -rf /tmp/vpc-build && mkdir -p /tmp/vpc-build
cp "$SRC/Dockerfile" "$SRC/pyproject.toml" "$SRC/uv.lock" "$SRC/README.md" "$SRC/.dockerignore" /tmp/vpc-build/
cp -R "$SRC/app" /tmp/vpc-build/app
find /tmp/vpc-build -name __pycache__ -type d -prune -exec rm -rf {} +

az acr login -n ca72f0c87afdacr
docker buildx create --name vpcbuilder --driver docker-container --use 2>/dev/null || docker buildx use vpcbuilder
docker buildx build --platform linux/amd64 \
  -t ca72f0c87afdacr.azurecr.io/visiting-place-clusterer:v1 \
  --push /tmp/vpc-build
```

## Create the Container App (first time)

```bash
ACR_PWD=$(az acr credential show -n ca72f0c87afdacr --query "passwords[0].value" -o tsv)
DOMAIN=$(az containerapp env show -n polimi-cloud-env -g polimi-cloud --query properties.defaultDomain -o tsv)

az containerapp create \
  --name visiting-place-clusterer -g polimi-cloud \
  --environment polimi-cloud-env \
  --image ca72f0c87afdacr.azurecr.io/visiting-place-clusterer:v1 \
  --target-port 8000 --ingress external \
  --registry-server ca72f0c87afdacr.azurecr.io \
  --registry-username ca72f0c87afdacr --registry-password "$ACR_PWD" \
  --min-replicas 1 --max-replicas 1 --cpu 0.5 --memory 1.0Gi \
  --secrets azure-openai-key="<AZURE_OPENAI_API_KEY>" \
  --env-vars APP_MODE=a2a PROVIDER=azure \
    AZURE_OPENAI_ENDPOINT=https://visiting-place-clusterer.openai.azure.com \
    AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini \
    AZURE_OPENAI_API_VERSION=2024-12-01-preview \
    AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
    "PUBLIC_URL=https://visiting-place-clusterer.$DOMAIN"
```

## Redeploy after code changes

```bash
# 1. rebuild + push a new tag (repeat the staging + buildx step with :v2)
docker buildx build --platform linux/amd64 \
  -t ca72f0c87afdacr.azurecr.io/visiting-place-clusterer:v2 --push /tmp/vpc-build

# 2. roll the app to the new image
az containerapp update -n visiting-place-clusterer -g polimi-cloud \
  --image ca72f0c87afdacr.azurecr.io/visiting-place-clusterer:v2
```

To change an env var or rotate the key:

```bash
az containerapp secret set -n visiting-place-clusterer -g polimi-cloud \
  --secrets azure-openai-key="<NEW_KEY>"
az containerapp update -n visiting-place-clusterer -g polimi-cloud \
  --set-env-vars AZURE_OPENAI_DEPLOYMENT=<deployment>
```

## Verify

```bash
BASE=https://visiting-place-clusterer.icypond-686fddf2.uksouth.azurecontainerapps.io
curl "$BASE/health"                          # -> ok
curl "$BASE/.well-known/agent-card.json"     # -> agent card JSON
# Logs:
az containerapp logs show -n visiting-place-clusterer -g polimi-cloud --tail 50
```

## Hardening (optional, not yet applied)

- **Keyless model auth:** give the Container App a managed identity, grant it the
  `Cognitive Services OpenAI User` role on the Foundry account, and switch
  `llm_model_service.py` to build the `AzureProvider` from an
  `AsyncAzureOpenAI(azure_ad_token_provider=...)` client (drop `AZURE_OPENAI_API_KEY`).
- **Keyless ACR pull:** use `--registry-identity system` instead of admin creds.
- **Rotate** the Foundry API key — it was shared in plaintext during setup.
