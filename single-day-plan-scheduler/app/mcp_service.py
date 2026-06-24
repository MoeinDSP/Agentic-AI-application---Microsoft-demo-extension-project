from __future__ import annotations

from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

from environment_service import env


class McpService:
    """Creates MCP toolsets backed by environment configuration.

    Unlike the other agents, which connect to the shared Google Maps MCP service,
    this agent runs its own dedicated MCP server (the single-day-plan-scheduler-mcp
    companion service) and connects to it here.
    """

    @staticmethod
    def create_all() -> list[McpToolset]:
        """Return every MCP toolset registered for the application."""
        return [McpService.routing()]

    @staticmethod
    def routing() -> McpToolset:
        """Create an MCP toolset for this agent's own routing/maps MCP server."""
        return McpToolset(
            connection_params=StreamableHTTPConnectionParams(url=env.mcp_url),
        )
