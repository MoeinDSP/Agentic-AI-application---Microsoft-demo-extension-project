from __future__ import annotations

from pydantic_ai.mcp import MCPServerStreamableHTTP

from config import settings

_GOOGLE_MAPS_MCP_URL = "https://mapstools.googleapis.com/mcp"


def create_google_maps_mcp() -> MCPServerStreamableHTTP:
    """Create a streamable-HTTP MCP client for the Google Maps API."""
    return MCPServerStreamableHTTP(
        url=_GOOGLE_MAPS_MCP_URL,
        headers={"X-Goog-Api-Key": settings.google_maps_api_key},
    )
