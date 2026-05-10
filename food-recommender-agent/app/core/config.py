from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM (OpenRouter via LiteLLM) ──────────────────────────────────────────
    # OpenRouter gives access to many providers behind one API key.
    # The model string MUST start with "openrouter/" — that's the prefix
    # LiteLLM uses to route the call. Examples:
    #   openrouter/openai/gpt-4o-mini
    #   openrouter/anthropic/claude-3.5-sonnet
    #   openrouter/google/gemini-2.0-flash-001
    #   openrouter/meta-llama/llama-3.3-70b-instruct
    openrouter_api_key: str
    openrouter_model: str = "openrouter/openai/gpt-4o-mini"

    # Optional but recommended by OpenRouter — shown on their dashboard,
    # used for usage attribution. Leave blank if you don't have one.
    openrouter_site_url: str = ""
    openrouter_app_name: str = "Food Recommender Agent"

    # ── Google Places API ─────────────────────────────────────────────────────
    # A simple API key (NOT a service account). See README for setup.
    google_places_api_key: str

    # ── A2A server ────────────────────────────────────────────────────────────
    agent_host: str = "0.0.0.0"
    agent_port: int = 8004
    agent_name: str = "Food Recommender Agent"
    agent_description: str = (
        "Precision-oriented restaurant recommendation agent. "
        "Given a meal slot, location, radius, budget, and preferences, "
        "returns a ranked list of RestaurantCandidate objects."
    )
    agent_url: str = "http://localhost:8004"
    agent_version: str = "1.0.0"

    # ── Agent 3 remote address ────────────────────────────────────────────────
    agent3_url: str = "http://localhost:8003"


settings = Settings()
