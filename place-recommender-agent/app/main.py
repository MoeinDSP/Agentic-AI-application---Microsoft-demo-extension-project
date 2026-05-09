from __future__ import annotations

import logfire
import uvicorn

from agent import AgentFactory
from config import AgentEnum, ModelEnum, ProviderEnum, settings
from llm import create_model
from mcps import create_google_maps_mcp

logfire.configure(token=settings.logfire_token)
logfire.instrument_pydantic_ai()

model = create_model(ProviderEnum.OPENAI, ModelEnum.GPT_5_4_MINI.value)

google_maps_mcp = create_google_maps_mcp()

agent_factory = AgentFactory(
    agent_enum=AgentEnum.PLACE_RECOMMENDER,
    model=model,
    toolsets=[google_maps_mcp],
)


if __name__ == "__main__":
    uvicorn.run(agent_factory.web, host="0.0.0.0", port=8000)
