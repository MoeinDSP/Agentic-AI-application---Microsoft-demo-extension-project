from __future__ import annotations

from pydantic_ai.mcp import MCPToolset

from environment_service import env


class McpService:
    """Creates MCP services backed by environment configuration."""

    @staticmethod
    def create_all() -> list[MCPToolset]:
        """Return every MCP service registered for the application."""
        return [McpService.google_maps()]

    @staticmethod
    def google_maps() -> MCPToolset:
        """Create an MCP service for the Google Maps API."""
        return MCPToolset(
            env.google_maps_mcp_url,
            headers={"X-Goog-Api-Key": env.google_maps_api_key},
        )
