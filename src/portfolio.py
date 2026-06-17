# src/portfolio.py

import numpy as np
import pandas as pd


def build_price_matrix(
    selected_assets: list[str],
    asset_tickers: dict[str, str],
    periodo: str,
    data_loader_func
) -> pd.DataFrame:
    """
    Construye una matriz de precios ajustados para múltiples activos.
    """

    price_data = {}

    for asset in selected_assets:
        ticker = asset_tickers[asset]
        prices = data_loader_func(ticker, periodo)

        if prices is not None and len(prices) > 0:
            price_data[asset] = prices

    prices_df = pd.DataFrame(price_data).dropna()

    return prices_df


def calculate_portfolio_returns(
    prices_df: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:
    """
    Calcula retornos diarios del portafolio.
    """

    returns = prices_df.pct_change().dropna()
    portfolio_returns = returns @ weights

    return portfolio_returns


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calcula máximo drawdown porcentual.
    """

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    return float(drawdown.min() * 100)


def calculate_portfolio_metrics(
    prices_df: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float = 10_000.0,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calcula métricas principales de portafolio.
    """

    if prices_df.empty or len(prices_df.columns) < 2:
        return {
            "available": False,
            "message": "Se requieren al menos dos activos con datos válidos."
        }

    returns = prices_df.pct_change().dropna()
    portfolio_returns = returns @ weights

    cumulative_return = (1 + portfolio_returns).prod() - 1

    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)

    if annualized_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (
            annualized_return - risk_free_rate
        ) / annualized_volatility

    equity_curve = initial_capital * (1 + portfolio_returns).cumprod()

    max_drawdown = calculate_max_drawdown(equity_curve)

    correlation_matrix = returns.corr()

    return {
        "available": True,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "equity_curve": equity_curve,
        "correlation_matrix": correlation_matrix,
        "cumulative_return": round(cumulative_return * 100, 2),
        "annualized_return": round(annualized_return * 100, 2),
        "annualized_volatility": round(annualized_volatility * 100, 2),
        "sharpe_ratio": round(float(sharpe_ratio), 2),
        "max_drawdown": round(max_drawdown, 2),
    }


def equal_weight_vector(n_assets: int) -> np.ndarray:
    """
    Genera pesos iguales para n activos.
    """

    if n_assets <= 0:
        return np.array([])

    return np.repeat(1 / n_assets, n_assets)
