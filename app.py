import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from alphagraph.pipeline import run  # noqa: E402  (must follow load_dotenv)

st.set_page_config(
    page_title="AlphaGraph — Financial Analysis Agent",
    page_icon="📈",
    layout="wide",
)

RECOMMENDATION_COLORS = {
    "BUY": ("green", "🟢"),
    "HOLD": ("orange", "🟡"),
    "SELL": ("red", "🔴"),
}


def _fmt(val, suffix=""):
    if val is None:
        return "N/A"
    return f"{val:.2f}{suffix}"


def render_metrics_table(metrics: dict, title: str):
    if not metrics or "error" in metrics:
        st.warning(f"Could not compute {title}.")
        return
    rows = {k.replace("_", " ").title(): _fmt(v, "%" if "pct" in k else "") for k, v in metrics.items()}
    df = pd.DataFrame(rows.items(), columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_growth_chart(income_stmt: dict, ticker: str):
    try:
        df = pd.DataFrame(income_stmt)
        if "Total Revenue" not in df.index:
            return
        revenue = df.loc["Total Revenue"].sort_index()
        net_income = df.loc["Net Income"].sort_index() if "Net Income" in df.index else None

        years = [str(d)[:4] for d in revenue.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=years,
            y=(revenue.values / 1e9).tolist(),
            name="Revenue (B)",
            marker_color="#4A90D9",
        ))
        if net_income is not None:
            fig.add_trace(go.Scatter(
                x=years,
                y=(net_income.values / 1e9).tolist(),
                name="Net Income (B)",
                mode="lines+markers",
                marker_color="#F5A623",
                yaxis="y2",
            ))
        fig.update_layout(
            title=f"{ticker} — Revenue & Net Income Trend",
            yaxis=dict(title="Revenue (USD Billions)"),
            yaxis2=dict(title="Net Income (USD Billions)", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def render_margin_chart(profitability: dict, ticker: str):
    if not profitability or "error" in profitability:
        return
    labels = []
    values = []
    for k in ["gross_margin_pct", "net_margin_pct", "ebit_margin_pct", "roe_pct", "roa_pct"]:
        if profitability.get(k) is not None:
            labels.append(k.replace("_pct", "").replace("_", " ").upper())
            values.append(profitability[k])
    if not labels:
        return
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in values],
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{ticker} — Profitability Metrics (%)",
        yaxis_title="Percentage (%)",
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("AlphaGraph")
st.sidebar.caption("Multi-Agent Financial Research System")
st.sidebar.markdown("---")
_raw_ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    max_chars=20,
    placeholder="e.g. AAPL, AMZN, MSFT",
    help="Enter a stock ticker symbol (not the company name).",
).upper().strip()
# Forgiving fallback: map common company names to their ticker symbols.
NAME_TO_TICKER = {
    "APPLE": "AAPL",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "MICROSOFT": "MSFT",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
}
ticker_input = NAME_TO_TICKER.get(_raw_ticker, _raw_ticker)
if ticker_input != _raw_ticker:
    st.sidebar.caption(f"Using ticker **{ticker_input}** for {_raw_ticker.title()}.")
run_btn = st.sidebar.button("Analyze", type="primary", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("""
1. Fetches financial statements via yfinance
2. Computes profitability, growth & liquidity ratios
3. Retrieves recent news via RAG
4. Assesses risk with LLM
5. Generates investment memo
""")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("AlphaGraph — Financial Analysis Agent")

if not run_btn:
    st.info("Enter a ticker in the sidebar and click **Analyze** to generate an investment memo.")
    st.stop()

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

with st.spinner(f"Running AlphaGraph pipeline for **{ticker_input}**…"):
    result = run(ticker_input)

if result.get("error"):
    st.error(f"Pipeline error: {result['error']}")
    st.stop()

company_name = result.get("company_name", ticker_input)
recommendation = result.get("recommendation", "HOLD")
color, icon = RECOMMENDATION_COLORS.get(recommendation, ("gray", "⚪"))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"## {company_name} ({ticker_input})")
rec_col, _, _ = st.columns([1, 3, 3])
with rec_col:
    st.markdown(
        f"<div style='background:{color};color:white;padding:12px 20px;"
        f"border-radius:8px;text-align:center;font-size:1.4rem;font-weight:bold'>"
        f"{icon} {recommendation}</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    if result.get("income_stmt"):
        render_growth_chart(result["income_stmt"], ticker_input)
with chart_col2:
    if result.get("profitability_metrics"):
        render_margin_chart(result["profitability_metrics"], ticker_input)

# ── Metrics Tables ────────────────────────────────────────────────────────────
st.markdown("### Quantitative Metrics")
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**Profitability**")
    render_metrics_table(result.get("profitability_metrics"), "Profitability")
with m2:
    st.markdown("**Growth**")
    render_metrics_table(result.get("growth_metrics"), "Growth")
with m3:
    st.markdown("**Liquidity & Leverage**")
    render_metrics_table(result.get("liquidity_metrics"), "Liquidity")

# ── News Signals ──────────────────────────────────────────────────────────────
st.markdown("### News Signals")
signals = result.get("news_signals", {"bullish": [], "bearish": []})
ns_col1, ns_col2 = st.columns(2)
with ns_col1:
    st.markdown("**Bullish**")
    if signals.get("bullish"):
        for s in signals["bullish"]:
            st.markdown(f"🟢 {s}")
    else:
        st.caption("None identified")
with ns_col2:
    st.markdown("**Bearish**")
    if signals.get("bearish"):
        for s in signals["bearish"]:
            st.markdown(f"🔴 {s}")
    else:
        st.caption("None identified")

# ── Investment Memo ───────────────────────────────────────────────────────────
st.markdown("### Investment Memo")
with st.expander("View full memo", expanded=True):
    st.text(result.get("investment_memo", "No memo generated."))
