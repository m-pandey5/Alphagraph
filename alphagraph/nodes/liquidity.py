import pandas as pd
from alphagraph.state import FinancialState


def _safe_div(a, b):
    try:
        if b and float(b) != 0:
            return round(float(a) / float(b), 2)
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


def liquidity_analyst(state: FinancialState) -> dict:
    if state.get("error"):
        return {}

    balance = pd.DataFrame(state["balance_sheet"])

    try:
        current_assets = _latest(balance, "Current Assets")
        current_liabilities = _latest(balance, "Current Liabilities")
        inventory = _latest(balance, "Inventory") or 0
        total_assets = _latest(balance, "Total Assets")
        total_debt = _latest(balance, "Total Debt")
        cash = _latest(balance, "Cash And Cash Equivalents")

        quick_assets = (float(current_assets or 0) - float(inventory))

        metrics = {
            "current_ratio": _safe_div(current_assets, current_liabilities),
            "quick_ratio": _safe_div(quick_assets, current_liabilities),
            "debt_to_assets": _safe_div(total_debt, total_assets),
            "cash_position_bn": round(float(cash) / 1e9, 2) if cash else None,
        }
        return {"liquidity_metrics": metrics}
    except Exception as e:
        return {"liquidity_metrics": {"error": str(e)}}
