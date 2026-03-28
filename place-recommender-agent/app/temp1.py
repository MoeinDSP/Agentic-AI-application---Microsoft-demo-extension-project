from strands import Agent, tool
from strands.agent.a2a_agent import A2AAgent
from strands.models.openai import OpenAIModel
import logfire

# logfire.configure()
# logfire.instrument_openai()


google_maps_agent = A2AAgent(
    endpoint="http://0.0.0.0:8000",
    name="google-agent"
)

@tool
def google_maps(expression: str) -> str:
    """Perform google maps operations."""
    result = google_maps_agent(expression)
    return str(result.message["content"][0]["text"])


model=OpenAIModel(
    client_args={
        'base_url': 'https://api.cerebras.ai/v1',
        'api_key': 'csk-v95hkc9tv4e6e5eky2j62wc3ffkktjfw4phy93efwm4dcp6f',
    },
    model_id='qwen-3-235b-a22b-instruct-2507'
)

orchestrator = Agent(
    model=model,
    system_prompt="You are a helpful assistant. Use your toogoogle_maps tool to respond to the users.",
    tools=[google_maps]
)

orchestrator('use the tool that is available to you, to recommend 5 places in milan, italy')
