import logfire
from fasta2a.schema import Skill
from dataclasses import replace
from pydantic_ai import Agent, ModelProfile, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.cerebras import CerebrasModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.providers.cerebras import CerebrasProvider

logfire.configure()
logfire.instrument_pydantic_ai()


# --- Model ---

class CerebrasCompatibleJsonSchemaTransformer(OpenAIJsonSchemaTransformer):
    def transform(self, schema):
        schema = super().transform(schema)
        if schema.get('type') in ('number', 'integer'):
            schema.pop('format', None)
        return schema


model = CerebrasModel(
    model_name='qwen-3-235b-a22b-instruct-2507',
    provider=CerebrasProvider(
        api_key=''
    ),
    profile=ModelProfile(
        json_schema_transformer=CerebrasCompatibleJsonSchemaTransformer,
    ),
)


# --- MCP ---

server = MCPServerStreamableHTTP(
    url='https://mapstools.googleapis.com/mcp',
    headers={
        "X-Goog-Api-Key": ""
    },
)


# --- Agent ---

SYSTEM_PROMPT = """
You are a helpful place recommender agent.
Your task is to recommend places to visit in a given city based on the user's accomodation location on that city.
If the user doesn't provide the city, or doesn't provide the accomodation location, you must ask for it.
Then using the city and the accomodation location, you must search for a circule of 2 kilometers around the accomodation location of the user.
For the place, if the user didn't provide the category, just search for visiting places in general,
but they provided some preferences, you must put them into the query while searching for the places.
You have the google maps MCP to search for the places. Use it with using the fully qualified name of the MCP.
Use as much as possible parameters it has.
"""


async def force_strict_tool_args(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Force stricter argument schemas so the model fills tool arguments."""
    forced_tool_defs: list[ToolDefinition] = []
    for tool_def in tool_defs:
        schema = dict(tool_def.parameters_json_schema or {})
        properties = schema.get('properties') or {}
        if properties:
            required = list(properties.keys())
            schema['required'] = required
            schema['additionalProperties'] = False

        forced_tool_defs.append(
            replace(
                tool_def,
                strict=True,
                parameters_json_schema=schema,
            )
        )

    return forced_tool_defs


agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    toolsets=[server],
    # prepare_tools=force_strict_tool_args,
    # retries=3,
)


# --- A2A ---

app = agent.to_a2a(
    name='Place Recommender Agent',
    url='http://0.0.0.0:8000',
    version='0.1.0',
    description='',
    skills=[
        Skill(
            id='place-recommendations',
            name='Place Recommendations',
            description=(
                'Recommends curated places to visit in a given city based on the user\'s preferences '
                'and constraints. Uses real-time Google Maps data to surface relevant landmarks, '
                'restaurants, parks, museums, and more. Supports filtering by interest category, '
                'travel style (solo, family, couple), budget level, and accessibility needs.'
            ),
            tags=[
                'travel',
                'tourism',
                'maps',
                'city-guide',
                'recommendations',
                'google-maps',
                'places',
                'itinerary',
            ],
            input_modes=['text'],
            output_modes=['text'],
            examples=[
                'Recommend places to visit in Italy, Milan',
                'Recommend places to visit in Spain, Barcelona that are related to architecture',
                'Recommend places to visit in France, Paris that are family-friendly and outdoors',
                'Recommend places to visit in Japan, Tokyo suitable for a solo traveler on a budget',
                'Recommend hidden gems to visit in Germany, Berlin',
            ],
        ),
    ],
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
