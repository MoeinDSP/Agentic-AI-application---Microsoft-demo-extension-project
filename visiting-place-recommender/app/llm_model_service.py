from __future__ import annotations

from pydantic_ai.models import Model, ModelProfile
from pydantic_ai.models.cerebras import CerebrasModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from environment_service import ModelEnum, ProviderEnum, env


class _CerebrasJsonSchemaTransformer(OpenAIJsonSchemaTransformer):
    """Strips 'format' from numeric JSON schema fields, which Cerebras does not support."""

    def transform(self, schema: dict) -> dict:
        schema = super().transform(schema)
        if schema.get("type") in ("number", "integer"):
            schema.pop("format", None)
        return schema


class LlmModelService:
    """Builds pydantic-ai models for configured providers."""

    @staticmethod
    def create(provider: ProviderEnum, model: ModelEnum) -> Model:
        """Build and return a pydantic-ai Model for the given provider and model name."""
        model_name = model.value

        match provider:
            case ProviderEnum.OPENAI:
                return OpenAIResponsesModel(
                    model_name=model_name,
                    provider=OpenAIProvider(api_key=env.openai_api_key),
                )
            case ProviderEnum.OPENROUTER:
                return OpenRouterModel(
                    model_name=model_name,
                    provider=OpenRouterProvider(api_key=env.openrouter_api_key),
                )
            case ProviderEnum.CEREBRAS:
                return CerebrasModel(
                    model_name=model_name,
                    provider=CerebrasProvider(api_key=env.cerebras_api_key),
                    profile=ModelProfile(
                        json_schema_transformer=_CerebrasJsonSchemaTransformer
                    ),
                )
            case _:
                raise ValueError(f"Unsupported LLM provider: {provider.value}")
