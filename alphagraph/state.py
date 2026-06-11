from typing import Optional
from typing_extensions import TypedDict


class FinancialState(TypedDict):
    # Input
    ticker: str
    company_name: str

    # Raw data from yfinance (stored as dicts for JSON serializability)
    income_stmt: Optional[dict]
    balance_sheet: Optional[dict]
    cash_flow: Optional[dict]

    # Pure-Python computed metrics (Nodes 2–4)
    profitability_metrics: Optional[dict]
    growth_metrics: Optional[dict]
    liquidity_metrics: Optional[dict]

    # LLM outputs (Nodes 5–7)
    news_signals: Optional[dict]   # {"bullish": [...], "bearish": [...]}
    risk_summary: Optional[str]
    investment_memo: Optional[str]
    recommendation: Optional[str]  # "BUY" | "HOLD" | "SELL"

    # Internal: raw news articles retrieved for Node 5
    raw_news: Optional[list[dict]]

    # Error propagation — any node sets this to abort gracefully
    error: Optional[str]
