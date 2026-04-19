from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Azure OpenAI (Azure AI Foundry) ──────────────────────────────────────
    # Set AZURE_OPENAI_ENDPOINT to activate the Azure backend.
    # Leave blank to use the OpenAI-compatible fallback below.
    azure_openai_endpoint: str = ""            # e.g. https://<resource>.openai.azure.com
    azure_openai_api_key: str = ""             # leave blank to use Managed Identity
    azure_openai_deployment: str = "gpt-4o-mini"   # your Azure deployment name
    azure_openai_api_version: str = "2024-12-01-preview"

    # ── OpenAI-compatible fallback (local dev / OpenRouter) ──────────────────
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # ── A2A server (this agent) ───────────────────────────────────────────────
    agent_host: str = "0.0.0.0"
    agent_port: int = 8001
    agent_name: str = "Visiting Place Clustering Agent"
    agent_description: str = (
        "Groups recommended places into geographically coherent day-clusters. "
        "Each cluster represents places intended for one trip day, "
        "minimizing travel distance within a day."
    )
    agent_url: str = "http://localhost:8001"
    agent_version: str = "1.0.0"

    # ── Tool-call loop ────────────────────────────────────────────────────────
    max_tool_rounds: int = 5


settings = Settings()
