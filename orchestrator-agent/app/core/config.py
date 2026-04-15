from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenRouter (OpenAI-compatible) ────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemini-2.0-flash-001"
    openrouter_site_url: str = ""       # optional — for OpenRouter leaderboard
    openrouter_app_name: str = "TripPlannerOrchestrator"

    # ── A2A server (this agent) ───────────────────────────────────────────────
    agent_host: str = "0.0.0.0"
    agent_port: int = 8080
    agent_name: str = "Trip Planner Orchestrator"
    agent_description: str = (
        "Main orchestrator for the trip planner system. "
        "Receives a user trip request (free-text or JSON), coordinates "
        "downstream agents (place recommender, clustering, daily scheduler, "
        "food recommender), and returns a complete daily itinerary."
    )
    agent_url: str = "http://localhost:8080"
    agent_version: str = "1.0.0"

    # ── Downstream agent addresses ────────────────────────────────────────────
    agent1_url: str = "http://localhost:8000"   # Place Recommender (A2A)
    agent2_url: str = "http://localhost:8001"   # Clustering (REST / FastAPI)
    agent4_url: str = "http://localhost:8004"   # Food Recommender (A2A)

    # ── Budget defaults ───────────────────────────────────────────────────────
    activity_budget_ratio: float = 0.70
    food_budget_ratio: float = 0.30


settings = Settings()
