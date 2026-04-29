from __future__ import annotations

import logfire
import uvicorn

from agent import AgentFactory
from config import AgentEnum, ModelEnum, ProviderEnum, settings
from llm import create_model
from mcps import create_google_maps_mcp

logfire.configure(token=settings.logfire_token)
logfire.instrument_pydantic_ai()

agent_factory = AgentFactory(
    agent_enum=AgentEnum.PLACE_RECOMMENDER,
    model=create_model(ProviderEnum.OPENROUTER, ModelEnum.OPENAI_GPT_OSS_120B_FREE.value),
    toolsets=[create_google_maps_mcp()],
)


if __name__ == "__main__":
    uvicorn.run(agent_factory.web, host="0.0.0.0", port=8000)
