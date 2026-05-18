from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic.fields import Field
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


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
    # Openai models
    GPT_5_4_MINI = "gpt-5.4-mini"

    # Openrouter models
    OPENAI_GPT_OSS_120B_FREE = "openai/gpt-oss-120b:free"

    # Cerebras models
    QWEN_3_235B_A22B_INSTRUCT_2507 = "qwen-3-235b-a22b-instruct-2507"


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


# Agent output types
class Place(BaseModel):
    name: str = Field(
        description="The name of the place as it appears on Google Maps."
    )
    place_url: str = Field(
        description="A Google Maps link to the place's main listing."
    )
    photos_url: str = Field(
        description="A Google Maps link to photos of the place."
    )
    reviews_url: str = Field(
        description="A Google Maps link to reviews of the place."
    )
    lat: float = Field(
        description="The latitude in degrees, in the range [-90.0, +90.0]."
    )
    lng: float = Field(
        description="The longitude in degrees, in the range [-180.0, +180.0]."
    )
    description: str = Field(
        description=(
            "A short rationale explaining why this place was selected and "
            "placed at this rank, grounded in the user's stated preferences, "
            "budget, and constraints."
        )
    )
    rank: int = Field(
        description=(
            "The position of the place in the recommendations list. Starts at "
            "0 for the highest-priority recommendation and increases by 1 with "
            "no gaps."
        ),
        ge=0,
    )


class PlaceRecommenderOutput(BaseModel):
    description: str = Field(
        description=(
            "A natural-language summary of the recommendation outcome. When "
            "`places` is non-empty, briefly describe the selection rationale. "
            "When `places` is empty, clearly state which required information "
            "is missing (e.g. destination city, type of experience) or why no "
            "suitable places were found."
        )
    )
    places: list[Place] = Field(
        default_factory=list,
        min_length=0,
        max_length=8,
        description=(
            "The ranked list of recommended places to visit in the destination "
            "city. Empty when required information is missing or no suitable "
            "matches were found."
        ),
    )


def get_output_type(agent: AgentEnum) -> BaseModel:
    match agent:
        case AgentEnum.PLACE_RECOMMENDER:
            return PlaceRecommenderOutput
        case _:
            raise ValueError(f"Invalid agent: {agent.value}")
