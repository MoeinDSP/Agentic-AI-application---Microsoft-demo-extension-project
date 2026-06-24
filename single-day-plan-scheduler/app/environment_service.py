"""
EnvironmentService class is the central unique point where we define
and store environment and configuration variable in our service.
That is important because everyone exactly knows where to find
information without creating different version of them.

The variables are divided according to their context, and they can be
read from `os.environ` or by other sources.
"""

from __future__ import annotations

from abc import ABCMeta
from enum import Enum
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentInitializationError(Exception):
    """Raised when environment variables fail validation during startup."""


class SingletonMeta(ABCMeta):
    """
    Singleton metaclass that creates a single instance of a class.
    This implementation is thread-safe.
    """

    _instances: dict["SingletonMeta", Any] = {}
    _locks: dict["SingletonMeta", Lock] = {}
    _global_lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._global_lock:
            if cls not in cls._locks:
                cls._locks[cls] = Lock()
        with cls._locks[cls]:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AppModeEnum(str, Enum):
    A2A = "a2a"
    WEB = "web"


class ProviderEnum(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"
    GOOGLE = "google"


class ModelEnum(str, Enum):
    GPT_5_4_MINI = "gpt-5.4-mini"
    OPENAI_GPT_4O_MINI_2024_07_18 = "openai/gpt-4o-mini-2024-07-18"
    OPENAI_GPT_OSS_120B_FREE = "openai/gpt-oss-120b:free"
    QWEN_3_235B_A22B_INSTRUCT_2507 = "qwen-3-235b-a22b-instruct-2507"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Application runtime settings."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    public_url: str = "http://localhost:8000"
    app_mode: AppModeEnum = AppModeEnum.A2A


class LlmSettings(BaseSettings):
    """LLM provider, model, and provider API keys."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    provider: ProviderEnum = ProviderEnum.OPENROUTER
    model_name: ModelEnum = ModelEnum.OPENAI_GPT_4O_MINI_2024_07_18
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    cerebras_api_key: str = ""
    google_api_key: str = ""


class McpSettings(BaseSettings):
    """Connection settings for this agent's own MCP server."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    mcp_url: str = "http://localhost:8005/mcp"


class FoodRecommenderSettings(BaseSettings):
    """A2A connection settings for the food-place-recommender agent."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    food_recommender_url: str = "http://localhost:8004"
    food_recommender_timeout_seconds: float = 10.0
    food_recommender_poll_interval_seconds: float = 1.0
    food_recommender_max_wait_seconds: float = 30.0


class ObservabilitySettings(BaseSettings):
    """Logging and tracing settings."""

    model_config = ConfigDict(frozen=True)

    logfire_token: str = ""


class EnvironmentVariables(BaseModel):
    """All environment-backed settings grouped by domain."""

    app: AppSettings
    llm: LlmSettings
    mcp: McpSettings
    food_recommender: FoodRecommenderSettings
    observability: ObservabilitySettings


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class EnvironmentService(metaclass=SingletonMeta):
    """
    This service is responsible for creating the environment variables.

    EnvironmentService is a singleton that when initialized saves all
    the variables in the class attribute `_env`.
    """

    _env: EnvironmentVariables

    def __init__(self) -> None:
        """Load and validate environment-backed settings."""

        try:
            self._env = EnvironmentVariables(
                app=AppSettings(),
                llm=LlmSettings(),
                mcp=McpSettings(),
                food_recommender=FoodRecommenderSettings(),
                observability=ObservabilitySettings(),
            )
        except ValidationError as e:
            raise EnvironmentInitializationError from e

        self._setup_logfire()

    def _setup_logfire(self) -> None:
        """Configure Logfire when a token is present."""
        if not self.logfire_token:
            return

        import logfire

        logfire.configure(token=self.logfire_token)

    # --- Application ---

    @property
    def public_url(self) -> str:
        """Return the public URL exposed by this service."""
        return self._env.app.public_url

    @property
    def app_mode(self) -> AppModeEnum:
        """Return the application serving mode (A2A or web UI)."""
        return self._env.app.app_mode

    # --- LLM ---

    @property
    def llm_provider(self) -> ProviderEnum:
        """Return the configured LLM provider."""
        return self._env.llm.provider

    @property
    def llm_model(self) -> ModelEnum:
        """Return the configured LLM model."""
        return self._env.llm.model_name

    @property
    def openai_api_key(self) -> str:
        """Return the OpenAI API key."""
        return self._env.llm.openai_api_key

    @property
    def openrouter_api_key(self) -> str:
        """Return the OpenRouter API key."""
        return self._env.llm.openrouter_api_key

    @property
    def cerebras_api_key(self) -> str:
        """Return the Cerebras API key."""
        return self._env.llm.cerebras_api_key

    @property
    def google_api_key(self) -> str:
        """Return the Google (Gemini) API key."""
        return self._env.llm.google_api_key

    # --- MCP ---

    @property
    def mcp_url(self) -> str:
        """Return the URL of this agent's own MCP server."""
        return self._env.mcp.mcp_url

    # --- Food recommender (A2A) ---

    @property
    def food_recommender_url(self) -> str:
        """Return the base A2A URL of the food-place-recommender agent."""
        return self._env.food_recommender.food_recommender_url

    @property
    def food_recommender_timeout_seconds(self) -> float:
        """Return the per-request timeout for food-place-recommender calls."""
        return self._env.food_recommender.food_recommender_timeout_seconds

    @property
    def food_recommender_poll_interval_seconds(self) -> float:
        """Return the A2A task poll interval for food-place-recommender calls."""
        return self._env.food_recommender.food_recommender_poll_interval_seconds

    @property
    def food_recommender_max_wait_seconds(self) -> float:
        """Return the maximum wait for a food-place-recommender A2A task."""
        return self._env.food_recommender.food_recommender_max_wait_seconds

    # --- Observability ---

    @property
    def logfire_token(self) -> str:
        """Return the Logfire token, if configured."""
        return self._env.observability.logfire_token


env = EnvironmentService()
