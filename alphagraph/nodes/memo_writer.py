import json
import os
from openai import OpenAI
from alphagraph.state import FinancialState

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MEMO_TEMPLATE = """
=== INVESTMENT MEMO ===
Company: {company_name} ({ticker})
Recommendation: {recommendation}

EXECUTIVE SUMMARY
{executive_summary}

QUANTITATIVE ANALYSIS
Profitability:
{profitability_section}

Growth:
{growth_section}

Liquidity & Leverage:
{liquidity_section}

QUALITATIVE ANALYSIS
Bullish Signals:
{bullish_signals}

Bearish Signals:
{bearish_signals}

RISK ASSESSMENT
{risk_summary}

INVESTMENT THESIS
{investment_thesis}

CAVEATS
{caveats}
""".strip()


def memo_writer(state: FinancialState) -> dict:
    if state.get("error"):
        return {}

    ticker = state["ticker"]
    company_name = state["company_name"]

    profitability = state.get("profitability_metrics", {})
    growth = state.get("growth_metrics", {})
    liquidity = state.get("liquidity_metrics", {})
    news_signals = state.get("news_signals", {"bullish": [], "bearish": []})
    risk_summary = state.get("risk_summary", "")

    prompt = f"""You are a senior investment analyst writing a professional investment memo for {company_name} ({ticker}).

Based on the data below, write a structured investment memo and make a final recommendation.

PROFITABILITY: {json.dumps(profitability)}
GROWTH: {json.dumps(growth)}
LIQUIDITY: {json.dumps(liquidity)}
BULLISH SIGNALS: {json.dumps(news_signals.get("bullish", []))}
BEARISH SIGNALS: {json.dumps(news_signals.get("bearish", []))}
RISK ASSESSMENT: {risk_summary}

Return a JSON object with these exact keys:
- "recommendation": one of "BUY", "HOLD", "SELL"
- "executive_summary": 2-3 sentence high-level summary
- "profitability_section": 2-3 sentences analyzing margins and returns
- "growth_section": 2-3 sentences analyzing growth trends
- "liquidity_section": 2-3 sentences analyzing balance sheet health
- "investment_thesis": 3-4 sentences explaining the core investment case
- "caveats": 2-3 important limitations or risks the investor should know

Be direct, specific, and cite numbers from the data where relevant."""

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    result = json.loads(response.choices[0].message.content)
    recommendation = result.get("recommendation", "HOLD")

    bullish_text = "\n".join(f"• {s}" for s in news_signals.get("bullish", [])) or "None identified"
    bearish_text = "\n".join(f"• {s}" for s in news_signals.get("bearish", [])) or "None identified"

    memo = MEMO_TEMPLATE.format(
        company_name=company_name,
        ticker=ticker,
        recommendation=recommendation,
        executive_summary=result.get("executive_summary", ""),
        profitability_section=result.get("profitability_section", ""),
        growth_section=result.get("growth_section", ""),
        liquidity_section=result.get("liquidity_section", ""),
        bullish_signals=bullish_text,
        bearish_signals=bearish_text,
        risk_summary=risk_summary,
        investment_thesis=result.get("investment_thesis", ""),
        caveats=result.get("caveats", ""),
    )

    return {
        "investment_memo": memo,
        "recommendation": recommendation,
    }
