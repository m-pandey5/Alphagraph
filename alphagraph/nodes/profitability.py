import pandas as pd
from alphagraph.state import FinancialState


def _safe_div(a, b):
    try:
        if b and b != 0:
            return round(float(a) / float(b) * 100, 2)
    except (TypeError, ValueError):
        pass
    return None


def _latest(df: pd.DataFrame, row: str):
    """Most recent value for a line item (rows are line items, columns are dates)."""
    if row in df.index:
        vals = df.loc[row].dropna()
        if len(vals):
            return float(vals.iloc[0])
    return None


def profitability_analyst(state: FinancialState) -> dict:
    if state.get("error"):
        return {}

    income = pd.DataFrame(state["income_stmt"])
    balance = pd.DataFrame(state["balance_sheet"])

    # Use most recent year (first column)
    try:
        total_revenue = _latest(income, "Total Revenue")
        gross_profit = _latest(income, "Gross Profit")
        net_income = _latest(income, "Net Income")
        ebit = _latest(income, "EBIT")

        total_assets = _latest(balance, "Total Assets") if not balance.empty else None
        stockholder_equity = _latest(balance, "Stockholders Equity") if not balance.empty else None

        metrics = {
            "gross_margin_pct": _safe_div(gross_profit, total_revenue),
            "net_margin_pct": _safe_div(net_income, total_revenue),
            "roe_pct": _safe_div(net_income, stockholder_equity),
            "roa_pct": _safe_div(net_income, total_assets),
            "ebit_margin_pct": _safe_div(ebit, total_revenue),
        }
        return {"profitability_metrics": metrics}
    except Exception as e:
        return {"profitability_metrics": {"error": str(e)}}
