"""MCP tool server exposing mock CI-support tools over stdio.

Adding a new tool is one function + one `@mcp.tool()` decorator here —
no other code changes required. FastMCP handles registration and
discovery, so `client.list_tools()` picks up new tools automatically.

Security boundary: tools only read from the local `data/` directory,
never accept file paths or shell input from the caller, and validate
identifier-shaped arguments (e.g. `run_id`) before using them as
lookup keys.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(__file__).parent / "data"
RUN_ID_RE = re.compile(r"^run-\d+$")

mcp = FastMCP("ci-triage-tools")


def _load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


@mcp.tool()
def fetch_build_artifact(run_id: str) -> dict:
    """Look up mock build-artifact metadata for a CI run_id (e.g. 'run-101')."""
    if not RUN_ID_RE.match(run_id):
        return {"error": f"invalid run_id format: {run_id!r}"}
    artifacts = _load("artifacts.json")
    return artifacts.get(run_id, {"error": f"no artifact metadata for {run_id}"})


@mcp.tool()
def lookup_test_owner(test_name: str) -> dict:
    """Look up the mock owning team for a test name (substring match)."""
    owners = _load("test_owners.json")
    for pattern, owner in owners.items():
        if pattern in test_name:
            return {"test_name": test_name, "owner": owner}
    return {"test_name": test_name, "owner": "unknown"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
