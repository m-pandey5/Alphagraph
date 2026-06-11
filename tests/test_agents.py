"""
Tests for pure-Python nodes (no API keys required).
"""

import pytest
from alphagraph.nodes.profitability import profitability_analyst
from alphagraph.nodes.growth import growth_analyst
from alphagraph.nodes.liquidity import liquidity_analyst


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_income_stmt():
    """Minimal income statement dict matching yfinance .to_dict() output."""
    return {
        "2023-09-30": {
            "Total Revenue": 383285000000,
            "Gross Profit": 169148000000,
            "Net Income": 96995000000,
            "EBIT": 114301000000,
            "Basic EPS": 6.16,
        },
        "2022-09-24": {
            "Total Revenue": 394328000000,
            "Gross Profit": 170782000000,
            "Net Income": 99803000000,
            "EBIT": 119437000000,
            "Basic EPS": 6.15,
        },
    }


@pytest.fixture
def sample_balance_sheet():
    return {
        "2023-09-30": {
            "Total Assets": 352583000000,
            "Current Assets": 143566000000,
            "Current Liabilities": 145308000000,
            "Stockholders Equity": 62146000000,
            "Total Debt": 111088000000,
            "Inventory": 6331000000,
            "Cash And Cash Equivalents": 29965000000,
        },
    }


@pytest.fixture
def base_state(sample_income_stmt, sample_balance_sheet):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "income_stmt": sample_income_stmt,
        "balance_sheet": sample_balance_sheet,
        "cash_flow": {},
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


# ── Profitability node ────────────────────────────────────────────────────────

def test_profitability_node_returns_dict(base_state):
    result = profitability_analyst(base_state)
    assert "profitability_metrics" in result
    assert isinstance(result["profitability_metrics"], dict)


def test_profitability_gross_margin(base_state):
    result = profitability_analyst(base_state)
    m = result["profitability_metrics"]
    # 169148 / 383285 * 100 ≈ 44.13
    assert m["gross_margin_pct"] == pytest.approx(44.13, abs=0.5)


def test_profitability_net_margin(base_state):
    result = profitability_analyst(base_state)
    m = result["profitability_metrics"]
    assert m["net_margin_pct"] == pytest.approx(25.31, abs=0.5)


def test_profitability_skips_on_error(base_state):
    base_state["error"] = "fetch failed"
    result = profitability_analyst(base_state)
    assert result == {}


# ── Growth node ───────────────────────────────────────────────────────────────

def test_growth_node_returns_dict(base_state):
    result = growth_analyst(base_state)
    assert "growth_metrics" in result
    assert isinstance(result["growth_metrics"], dict)


def test_growth_revenue_is_negative(base_state):
    result = growth_analyst(base_state)
    m = result["growth_metrics"]
    # Revenue fell from 394B to 383B
    assert m["revenue_growth_pct"] < 0


def test_growth_skips_on_error(base_state):
    base_state["error"] = "fetch failed"
    result = growth_analyst(base_state)
    assert result == {}


# ── Liquidity node ────────────────────────────────────────────────────────────

def test_liquidity_node_returns_dict(base_state):
    result = liquidity_analyst(base_state)
    assert "liquidity_metrics" in result
    assert isinstance(result["liquidity_metrics"], dict)


def test_liquidity_current_ratio(base_state):
    result = liquidity_analyst(base_state)
    m = result["liquidity_metrics"]
    # 143566 / 145308 ≈ 0.99
    assert m["current_ratio"] == pytest.approx(0.99, abs=0.05)


def test_liquidity_cash_position(base_state):
    result = liquidity_analyst(base_state)
    m = result["liquidity_metrics"]
    assert m["cash_position_bn"] == pytest.approx(29.97, abs=0.1)


def test_liquidity_skips_on_error(base_state):
    base_state["error"] = "fetch failed"
    result = liquidity_analyst(base_state)
    assert result == {}
