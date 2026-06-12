"""
MCP server exposing AlphaGraph as callable tools.

Lets any MCP client (Claude Desktop, Cursor, other agents) run the financial
analysis pipeline directly via tool calls.

Run locally:
    python -m alphagraph.mcp_server

Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "alphagraph": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["-m", "alphagraph.mcp_server"],
          "cwd": "/absolute/path/to/Financial_analyst"
        }
      }
    }
"""

import json

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from alphagraph.pipeline import run  # noqa: E402  (must follow load_dotenv)

mcp = FastMCP("AlphaGraph")


@mcp.tool()
def analyze_stock(ticker: str) -> str:
    """Run the full AlphaGraph pipeline for a stock ticker and return a complete
    investment memo with a Buy/Hold/Sell recommendation.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "AMZN", "NVDA".
    """
    result = run(ticker)
    if result.get("error"):
        return f"Error: {result['error']}"
    return result.get("investment_memo", "No memo generated.")


@mcp.tool()
def get_financial_metrics(ticker: str) -> str:
    """Return the computed profitability, growth, and liquidity ratios for a ticker
    as JSON. These are deterministic Python calculations (no LLM).

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    result = run(ticker)
    if result.get("error"):
        return f"Error: {result['error']}"
    return json.dumps(
        {
            "ticker": result.get("ticker"),
            "company_name": result.get("company_name"),
            "profitability": result.get("profitability_metrics"),
            "growth": result.get("growth_metrics"),
            "liquidity": result.get("liquidity_metrics"),
        },
        indent=2,
    )


@mcp.tool()
def get_news_signals(ticker: str) -> str:
    """Return bullish and bearish news signals for a ticker, retrieved from the
    FAISS news index and extracted by the LLM. Returns JSON.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    result = run(ticker)
    if result.get("error"):
        return f"Error: {result['error']}"
    return json.dumps(result.get("news_signals", {"bullish": [], "bearish": []}), indent=2)


@mcp.tool()
def get_recommendation(ticker: str) -> str:
    """Return just the Buy/Hold/Sell recommendation for a ticker (fast summary).

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    result = run(ticker)
    if result.get("error"):
        return f"Error: {result['error']}"
    return f"{result.get('ticker', ticker.upper())}: {result.get('recommendation', 'N/A')}"


if __name__ == "__main__":
    mcp.run()
