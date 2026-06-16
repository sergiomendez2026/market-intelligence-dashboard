# src/financial_metrics.py

import numpy as np
import pandas as pd


def cumulative_return(price: pd.Series) -> float:
    """
    Retorno acumulado del período.
    """
    if len(price) < 2:
        return 0.0

    return ((price.iloc[-1] / price.iloc[0]) - 1) * 100


def annualized_volatility(price: pd.Series, periods_per_year: int = 252) -> float:
    """
    Volatilidad anualizada usando retornos diarios.
    """
    returns = price.pct_change().dropna()

    if len(returns) == 0:
        return 0.0

    return returns.std() * np.sqrt(periods_per_year) * 100


def max_drawdown(price: pd.Series) -> float:
    """
    Máxima caída desde un máximo histórico local.
    """
    cumulative_max = price.cummax()
    drawdown = (price / cumulative_max) - 1

    return drawdown.min() * 100


def calculate_financial_kpis(price: pd.Series) -> dict:
    """
    Calcula KPIs financieros ejecutivos.
    """
    return {
        "cumulative_return": cumulative_return(price),
        "annualized_volatility": annualized_volatility(price),
        "max_drawdown": max_drawdown(price),
    }
