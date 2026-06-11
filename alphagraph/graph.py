from langgraph.graph import StateGraph, END
from alphagraph.state import FinancialState
from alphagraph.nodes.data_collector import financial_data_collector
from alphagraph.nodes.profitability import profitability_analyst
from alphagraph.nodes.growth import growth_analyst
from alphagraph.nodes.liquidity import liquidity_analyst
from alphagraph.nodes.news_research import news_research_agent
from alphagraph.nodes.risk_agent import risk_agent
from alphagraph.nodes.memo_writer import memo_writer


def _should_continue(state: FinancialState) -> str:
    """Abort the pipeline early if data collection failed."""
    return "end" if state.get("error") else "continue"


def build_graph():
    graph = StateGraph(FinancialState)

    graph.add_node("data_collector", financial_data_collector)
    graph.add_node("profitability", profitability_analyst)
    graph.add_node("growth", growth_analyst)
    graph.add_node("liquidity", liquidity_analyst)
    graph.add_node("news_research", news_research_agent)
    graph.add_node("risk", risk_agent)
    graph.add_node("memo", memo_writer)

    graph.set_entry_point("data_collector")

    # Conditional exit after data collection — bail out on bad tickers
    graph.add_conditional_edges(
        "data_collector",
        _should_continue,
        {"continue": "profitability", "end": END},
    )

    graph.add_edge("profitability", "growth")
    graph.add_edge("growth", "liquidity")
    graph.add_edge("liquidity", "news_research")
    graph.add_edge("news_research", "risk")
    graph.add_edge("risk", "memo")
    graph.add_edge("memo", END)

    return graph.compile()
