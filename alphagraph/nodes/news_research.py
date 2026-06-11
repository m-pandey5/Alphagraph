import json
import os
from openai import OpenAI
from alphagraph.state import FinancialState
from alphagraph.rag.retriever import retrieve

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def news_research_agent(state: FinancialState) -> dict:
    if state.get("error"):
        return {}

    ticker = state["ticker"]
    company_name = state["company_name"]

    query = f"{company_name} {ticker} financial performance earnings outlook"
    articles = retrieve(query, k=6, ticker=ticker)

    if not articles:
        return {
            "raw_news": [],
            "news_signals": {"bullish": [], "bearish": []},
        }

    news_text = "\n\n".join(
        f"[{i+1}] {a.get('title', '')}\n{a.get('content', '')}"
        for i, a in enumerate(articles)
    )

    prompt = f"""You are a financial news analyst. Read the following news articles about {company_name} ({ticker}) and extract investment signals.

Return a JSON object with exactly two keys:
- "bullish": list of strings, each a concise bullish signal found in the news (max 5)
- "bearish": list of strings, each a concise bearish signal found in the news (max 5)

If no signals of a type are found, return an empty list.

NEWS ARTICLES:
{news_text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    signals = json.loads(response.choices[0].message.content)
    return {
        "raw_news": articles,
        "news_signals": {
            "bullish": signals.get("bullish", []),
            "bearish": signals.get("bearish", []),
        },
    }
