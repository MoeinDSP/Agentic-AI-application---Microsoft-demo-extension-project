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
    openrouter_app_name: str = "TripPlannerOrchestrator"

    # ── A2A server (this agent) ───────────────────────────────────────────────
    agent_host: str = "0.0.0.0"
    agent_port: int = 8080
    agent_name: str = "Trip Planner Orchestrator"
    agent_description: str = (
        "Conversational trip planner. Chat naturally about your trip — "
        "the agent recommends places, clusters them by day, and builds "
        "a daily schedule with restaurants, all via downstream A2A agents."
    )
    agent_url: str = "http://localhost:8080"
    agent_version: str = "2.0.0"

    # ── Downstream agents (all A2A) ───────────────────────────────────────────
    agent1_url: str = "http://localhost:8000"   # Place Recommender
    agent2_url: str = "http://localhost:8001"   # Clustering
    agent3_url: str = "http://localhost:8003"   # Daily Scheduler (calls Agent 4)

    # ── Budget defaults ───────────────────────────────────────────────────────
    activity_budget_ratio: float = 0.70
    food_budget_ratio: float = 0.30

    # ── Tool-call loop ────────────────────────────────────────────────────────
    max_tool_rounds: int = 15


settings = Settings()
