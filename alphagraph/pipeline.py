"""
CLI entry point for running the AlphaGraph pipeline.

Usage:
    python -m alphagraph.pipeline --ticker AAPL
"""

import argparse
from dotenv import load_dotenv

load_dotenv()

from alphagraph.graph import build_graph  # noqa: E402  (must follow load_dotenv)


def run(ticker: str) -> dict:
    graph = build_graph()
    initial_state = {
        "ticker": ticker,
        "company_name": "",
        "income_stmt": None,
        "balance_sheet": None,
        "cash_flow": None,
        "profitability_metrics": None,
        "growth_metrics": None,
        "liquidity_metrics": None,
        "news_signals": None,
        "raw_news": None,
        "risk_summary": None,
        "investment_memo": None,
        "recommendation": None,
        "error": None,
    }
    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGraph — Financial Analysis Agent")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    print(f"\nRunning AlphaGraph for: {args.ticker.upper()}\n{'='*50}")
    result = run(args.ticker)

    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(result.get("investment_memo", "No memo generated."))
        print(f"\nRecommendation: {result.get('recommendation', 'N/A')}")
