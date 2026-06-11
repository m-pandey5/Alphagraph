import yfinance as yf
from alphagraph.state import FinancialState


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
            "income_stmt": income_stmt.to_dict(),
            "balance_sheet": balance_sheet.to_dict() if balance_sheet is not None else {},
            "cash_flow": cash_flow.to_dict() if cash_flow is not None else {},
            "error": None,
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for '{ticker}': {e}"}
