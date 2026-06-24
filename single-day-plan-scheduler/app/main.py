from __future__ import annotations

from agent_service import AgentEnum, AgentService
from environment_service import AppModeEnum, env
from food_recommender_service import recommend_restaurants
from llm_model_service import LlmModelService
from mcp_service import McpService
from starlette.types import ASGIApp, Receive, Scope, Send


class Application:
    """Wires services together and exposes the ASGI application."""

    @staticmethod
    def build() -> ASGIApp:
        """Return the configured ASGI app with health routing applied."""
        agent = AgentService.create(
            agent_enum=AgentEnum.PLAN_SCHEDULER,
            model=LlmModelService.create(env.llm_provider, env.llm_model),
            toolsets=McpService.create_all(),
            tools=[recommend_restaurants],
        )
        base_app: ASGIApp = agent.a2a if env.app_mode == AppModeEnum.A2A else agent.web
        return Application._add_health_route(base_app)

    @staticmethod
    def _add_health_route(asgi_app: ASGIApp) -> ASGIApp:
        """Wrap an ASGI app with a /health endpoint that always returns 200 ok."""

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            if scope.get("type") == "http" and scope.get("path") == "/health":
                await send(
                    {"type": "http.response.start", "status": 200, "headers": []}
                )
                await send({"type": "http.response.body", "body": b"ok"})
                return
            await asgi_app(scope, receive, send)

        return app


app = Application.build()
