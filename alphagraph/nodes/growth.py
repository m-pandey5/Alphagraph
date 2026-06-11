import pandas as pd
from alphagraph.state import FinancialState


def _yoy_growth(series: pd.Series) -> float | None:
    """Compute YoY growth between the two most recent periods."""
    vals = series.dropna().values
    if len(vals) < 2:
        return None
    current, prior = float(vals[0]), float(vals[1])
    if prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 2)


def growth_analyst(state: FinancialState) -> dict:
    if state.get("error"):
        return {}

    income = pd.DataFrame(state["income_stmt"])

    try:
        revenue_growth = _yoy_growth(income.loc["Total Revenue"]) if "Total Revenue" in income.index else None
        net_income_growth = _yoy_growth(income.loc["Net Income"]) if "Net Income" in income.index else None
        gross_profit_growth = _yoy_growth(income.loc["Gross Profit"]) if "Gross Profit" in income.index else None

        # EPS from income statement (Basic EPS)
        eps_growth = None
        for label in ["Basic EPS", "Diluted EPS", "EPS"]:
            if label in income.index:
                eps_growth = _yoy_growth(income.loc[label])
                break

        metrics = {
            "revenue_growth_pct": revenue_growth,
            "net_income_growth_pct": net_income_growth,
            "gross_profit_growth_pct": gross_profit_growth,
            "eps_growth_pct": eps_growth,
        }
        return {"growth_metrics": metrics}
    except Exception as e:
        return {"growth_metrics": {"error": str(e)}}
