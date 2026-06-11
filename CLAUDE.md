# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AlphaGraph** — a multi-agent financial research system built with LangGraph. Given a stock ticker, a supervisor orchestrates seven specialized agents to produce a structured investment memo with a Buy/Hold/Sell recommendation.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run the agent pipeline directly (CLI)
python -m alphagraph.pipeline --ticker AAPL

# Build/update the FAISS news index
python -m alphagraph.rag.build_index

# Run tests
pytest

# Run a single test file
pytest tests/test_agents.py

# Run a single test by name
pytest tests/test_agents.py::test_profitability_node -v

# Lint
ruff check .
ruff format .
```

## Architecture

### Agent Pipeline (`alphagraph/`)

Seven nodes wired into a LangGraph `StateGraph`. State flows sequentially through the graph; each node reads from and writes to a shared `FinancialState` TypedDict.

```
Supervisor
 ├── Node 1: financial_data_collector   — yfinance fetch (income stmt, balance sheet, cash flow)
 ├── Node 2: profitability_analyst      — pure Python: ROE, ROA, Net Margin, Gross Margin
 ├── Node 3: growth_analyst             — pure Python: Revenue/EPS/Profit growth YoY
 ├── Node 4: liquidity_analyst          — pure Python: Current Ratio, Quick Ratio, Debt Ratio
 ├── Node 5: news_research_agent        — RAG over news; outputs bullish/bearish signals (LLM)
 ├── Node 6: risk_agent                 — LLM: synthesizes debt, margins, news, sector risk
 └── Node 7: memo_writer                — LLM: final investment memo + Buy/Hold/Sell
```

Nodes 2–4 are pure-Python computation; they do **not** call the LLM. Nodes 5–7 call OpenAI.

### State Schema (`alphagraph/state.py`)

Single `FinancialState` TypedDict carries all data across nodes:
- `ticker`, `company_name`
- `income_stmt`, `balance_sheet`, `cash_flow` — raw DataFrames from yfinance
- `profitability_metrics`, `growth_metrics`, `liquidity_metrics` — dicts of computed ratios
- `news_signals` — `{"bullish": [...], "bearish": [...]}`
- `risk_summary` — string
- `investment_memo` — final structured string
- `recommendation` — `"BUY"` | `"HOLD"` | `"SELL"`

### RAG Setup (`alphagraph/rag/`)

- `build_index.py` — fetches recent news (via NewsAPI or similar), embeds with `sentence-transformers/all-MiniLM-L6-v2`, stores in FAISS on disk at `data/faiss_index/`
- `retriever.py` — loads the FAISS index and exposes a `retrieve(query, k=5)` function used by Node 5

### Streamlit UI (`app.py`)

Single-page app: ticker input → triggers the LangGraph pipeline → displays:
- Plotly charts (revenue growth, EPS trend, margin trend)
- Computed ratio tables
- News signals
- Final investment memo with recommendation badge

### Graph Construction (`alphagraph/graph.py`)

Builds and compiles the `StateGraph`. Entry point: `build_graph()` returns a compiled runnable. The supervisor is implemented as a simple sequential edge chain (not a dynamic router) for v1.

## Key Conventions

- Nodes 2–4 must remain LLM-free — all ratio logic is deterministic Python so it's testable without API keys.
- All OpenAI calls use structured output (JSON mode or response_format) so downstream nodes parse dicts, not free text.
- `FinancialState` is the single source of truth — nodes must not pass data via return values or globals.
- The FAISS index is pre-built and committed to `data/faiss_index/`; the app does not rebuild it at runtime.

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=
NEWS_API_KEY=        # for building the FAISS news index
```

Load via `python-dotenv` at startup.

## Tech Stack

| Layer | Library |
|---|---|
| Agent orchestration | `langgraph`, `langchain` |
| LLM | `openai` |
| Financial data | `yfinance` |
| RAG embeddings | `sentence-transformers` |
| Vector store | `faiss-cpu` |
| Visualization | `plotly` |
| UI | `streamlit` |
| Data | `pandas`, `numpy` |
| Linting | `ruff` |
| Tests | `pytest` |
