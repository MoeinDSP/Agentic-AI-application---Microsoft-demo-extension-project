from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="agent-3")
    environment: str = Field(default="development")
    service_version: str = Field(default="0.1.0")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8080)
    public_base_url: str = Field(default="http://127.0.0.1:8080")
    agent_card_path: str = Field(default="/.well-known/agent-card.json")
    a2a_path: str = Field(default="/a2a")
    mcp_base_url: str = Field(default="http://127.0.0.1:8090")
    mcp_timeout_seconds: float = Field(default=2.0)
    agent4_base_url: str = Field(default="http://127.0.0.1:8070")
    agent4_timeout_seconds: float = Field(default=2.0)
    fallback_travel_minutes: int = Field(default=0, ge=0)
    default_transport_mode: str = Field(default="walk")
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_prefix="AGENT3_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
