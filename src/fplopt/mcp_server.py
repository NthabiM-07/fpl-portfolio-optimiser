"""Model Context Protocol server: lets any MCP-capable AI assistant call the optimiser as a tool.

Start it with `python main.py mcp` (stdio transport); VS Code picks it up from `.vscode/mcp.json`.
Then ask: "Pick my GW 20 team under GBP 80m without Haaland, and show me the safer option."
"""
from __future__ import annotations

from . import assistant

INSTRUCTIONS = (
    "Tools for Fantasy Premier League team selection on the 2025-26 season. Every number comes from "
    "walk-forward forecasts and an exact integer programme; never estimate points yourself. Budgets are "
    "for the eleven starters in GBP millions (default 83.5). risk_gamma > 0 trades expected points for a "
    "better worst case."
)
TOOLS = [assistant.pick_team, assistant.explain_player, assistant.compare_risk, assistant.backtest_summary]


def build_server():
    try:                                            # mcp >= 2.0
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:                             # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server
    server = Server("fpl-portfolio-optimiser", instructions=INSTRUCTIONS)
    for fn in TOOLS:
        server.add_tool(fn)
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
