"""Thin MCP client: spawns the local tool server over stdio and exposes
list_tools()/call_many() as plain synchronous calls for agents.py.

Each call spawns one server subprocess and closes it when done — simple
and correct for a demo-scoped tool set; a long-lived session would be
the next step if tool-call volume grew.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SERVER_SCRIPT = Path(__file__).parent / "server.py"


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])


def _extract(result: Any) -> dict:
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    return {}


async def _run(coro_factory):
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await coro_factory(session)


def list_tools() -> list[dict]:
    async def go(session: ClientSession):
        result = await session.list_tools()
        return [{"name": t.name, "description": t.description} for t in result.tools]

    return asyncio.run(_run(go))


def call_many(calls: list[tuple[str, dict]]) -> list[dict]:
    """Run several tool calls against one spawned server session."""

    async def go(session: ClientSession):
        results = []
        for name, arguments in calls:
            result = await session.call_tool(name, arguments)
            results.append(_extract(result))
        return results

    return asyncio.run(_run(go))
