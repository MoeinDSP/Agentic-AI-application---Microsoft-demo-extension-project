from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mcps import MCPEnum, create_mcp


def _dump_tool(tool: Any) -> Any:
    if hasattr(tool, "model_dump"):
        return tool.model_dump(mode="json")
    if hasattr(tool, "dict"):
        return tool.dict()
    return vars(tool)


async def export_google_maps_tools(output_path: str = "google_maps_mcp_tools_full.json") -> list[Any]:
    google_maps_mcp = create_mcp(MCPEnum.GOOGLE_MAPS)

    async with google_maps_mcp:
        tools = await google_maps_mcp.list_tools()

    tools_json = [_dump_tool(tool) for tool in tools]

    Path(output_path).write_text(
        json.dumps(tools_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Stored {len(tools_json)} tools in {output_path}")
    return tools_json


if __name__ == "__main__":
    asyncio.run(export_google_maps_tools())
