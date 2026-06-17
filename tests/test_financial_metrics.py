import pandas as pd

from src.financial_metrics import calculate_financial_kpis


def test_financial_kpis_output_keys():
    prices = pd.Series([100, 102, 101, 105, 107, 106])

    result = calculate_financial_kpis(prices)

    expected_keys = {
        "cumulative_return",
        "annualized_volatility",
        "max_drawdown",
    }

    assert expected_keys.issubset(result.keys())


def test_cumulative_return_positive():
    prices = pd.Series([100, 110])

    result = calculate_financial_kpis(prices)

    assert result["cumulative_return"] > 0
