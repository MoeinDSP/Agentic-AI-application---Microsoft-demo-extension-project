from __future__ import annotations

from google.adk.models.lite_llm import LiteLlm

from environment_service import ModelEnum, ProviderEnum, env


class LlmModelService:
    """Builds Google ADK models for configured providers.

    Non-Google providers are reached through LiteLLM so the agent can use the
    same OpenRouter/OpenAI/Cerebras models as the other services in the project,
    while still running on the Google Agent Development Kit. Google (Gemini)
    models are passed through natively as a plain model-name string.
    """

    @staticmethod
    def create(provider: ProviderEnum, model: ModelEnum) -> LiteLlm | str:
        """Build and return an ADK model for the given provider and model name."""
        model_name = model.value

        match provider:
            case ProviderEnum.OPENROUTER:
                return LiteLlm(
                    model=f"openrouter/{model_name}",
                    api_key=env.openrouter_api_key,
                )
            case ProviderEnum.OPENAI:
                return LiteLlm(
                    model=f"openai/{model_name}",
                    api_key=env.openai_api_key,
                )
            case ProviderEnum.CEREBRAS:
                return LiteLlm(
                    model=f"cerebras/{model_name}",
                    api_key=env.cerebras_api_key,
                )
            case ProviderEnum.GOOGLE:
                return model_name
            case _:
                raise ValueError(f"Unsupported LLM provider: {provider.value}")
