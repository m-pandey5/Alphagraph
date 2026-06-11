import json
import os
from openai import OpenAI
from alphagraph.state import FinancialState

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def risk_agent(state: FinancialState) -> dict:
    if state.get("error"):
        return {}

    ticker = state["ticker"]
    company_name = state["company_name"]

    profitability = state.get("profitability_metrics", {})
    liquidity = state.get("liquidity_metrics", {})
    growth = state.get("growth_metrics", {})
    news_signals = state.get("news_signals", {"bullish": [], "bearish": []})

    prompt = f"""You are a financial risk analyst. Assess the investment risk for {company_name} ({ticker}).

Use the data below to identify the top risks. Be specific and concise.

PROFITABILITY METRICS:
{json.dumps(profitability, indent=2)}

GROWTH METRICS:
{json.dumps(growth, indent=2)}

LIQUIDITY METRICS:
{json.dumps(liquidity, indent=2)}

BEARISH NEWS SIGNALS:
{json.dumps(news_signals.get("bearish", []), indent=2)}

Return a JSON object with:
- "risk_level": one of "LOW", "MEDIUM", "HIGH"
- "key_risks": list of 3-5 specific risk factors (strings)
- "risk_summary": a 2-3 sentence narrative summary of the overall risk profile"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    result = json.loads(response.choices[0].message.content)
    risk_narrative = (
        f"Risk Level: {result.get('risk_level', 'UNKNOWN')}\n\n"
        f"Key Risks:\n" + "\n".join(f"• {r}" for r in result.get("key_risks", [])) +
        f"\n\n{result.get('risk_summary', '')}"
    )
    return {"risk_summary": risk_narrative}
