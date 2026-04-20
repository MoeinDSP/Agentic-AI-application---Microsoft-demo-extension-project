from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent3.models.agent4 import (
    AGENT4_INVOCATION_MODE_HTTP,
    SUPPORTED_AGENT4_INVOCATION_MODES,
)
from agent3.models.plan import DEFAULT_TRANSPORT_MODE, SUPPORTED_TRANSPORT_MODES


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
    agent4_invocation_mode: str = Field(default=AGENT4_INVOCATION_MODE_HTTP)
    fallback_travel_minutes: int = Field(default=0, ge=0)
    default_transport_mode: str = Field(default=DEFAULT_TRANSPORT_MODE)
    log_level: str = Field(default="INFO")

    @field_validator("agent4_invocation_mode")
    @classmethod
    def validate_agent4_invocation_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_AGENT4_INVOCATION_MODES:
            supported_modes = ", ".join(sorted(SUPPORTED_AGENT4_INVOCATION_MODES))
            raise ValueError(
                f"agent4_invocation_mode must be one of: {supported_modes}"
            )
        return normalized

    @field_validator("default_transport_mode")
    @classmethod
    def validate_default_transport_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_TRANSPORT_MODES:
            raise ValueError(
                "default_transport_mode must be one of: "
                "walking, driving, transit, bicycling"
            )
        return normalized

    model_config = SettingsConfigDict(
        env_prefix="AGENT3_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
