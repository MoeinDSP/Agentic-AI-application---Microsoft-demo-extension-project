from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


_FILES_DIR = Path(__file__).parent / "files"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openai_api_key: str = ""
    cerebras_api_key: str = ""
    openrouter_api_key: str = ""
    google_maps_api_key: str = ""
    logfire_token: str = ""


settings = Settings()


class AgentEnum(str, Enum):
    PLACE_RECOMMENDER = "place_recommender"


class ProviderEnum(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"


class ModelEnum(str, Enum):
    QWEN_3_235B_A22B_INSTRUCT_2507 = "qwen-3-235b-a22b-instruct-2507"
    OPENAI_GPT_OSS_120B_FREE = "openai/gpt-oss-120b:free"


@lru_cache(maxsize=None)
def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_system_prompt(agent: AgentEnum) -> str:
    data = _load_yaml(_FILES_DIR / "system_prompts.yml")
    return data[agent.value]


def get_agent_card(agent: AgentEnum) -> dict:
    data = _load_yaml(_FILES_DIR / "agent_cards.yml")
    return data[agent.value]
