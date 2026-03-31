from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Google Cloud ──────────────────────────────────────────────────────────
    google_cloud_project: str
    google_cloud_location: str = "us-central1"

    # ── Google Places API ─────────────────────────────────────────────────────
    google_places_api_key: str

    # ── Vertex AI / ADK ───────────────────────────────────────────────────────
    vertex_ai_model: str = "gemini-2.0-flash"

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