from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="agent-3-mcp")
    environment: str = Field(default="development")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8090)
    google_maps_api_key: str | None = Field(default=None)
    google_routes_timeout_seconds: float = Field(default=5.0, gt=0)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_prefix="AGENT3_MCP_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
