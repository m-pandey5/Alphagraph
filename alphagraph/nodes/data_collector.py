import pandas as pd
import yfinance as yf
from alphagraph.state import FinancialState


def _df_to_dict(df: pd.DataFrame | None) -> dict:
    """Serialize a yfinance statement to a JSON-clean dict.

    yfinance uses pandas Timestamp objects as column labels, which are not valid
    JSON keys (breaks LangSmith tracing / any JSON serialization). Stringify the
    date columns while preserving their order (most recent first).
    """
    if df is None or df.empty:
        return {}
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df.to_dict()


def financial_data_collector(state: FinancialState) -> dict:
    ticker = state["ticker"].upper().strip()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        company_name = info.get("longName") or info.get("shortName") or ticker

        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        if income_stmt is None or income_stmt.empty:
            return {"error": f"No financial data found for ticker '{ticker}'."}

        return {
            "ticker": ticker,
            "company_name": company_name,
            "income_stmt": _df_to_dict(income_stmt),
            "balance_sheet": _df_to_dict(balance_sheet),
            "cash_flow": _df_to_dict(cash_flow),
            "error": None,
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for '{ticker}': {e}"}
