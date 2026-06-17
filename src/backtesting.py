# src/backtesting.py

import numpy as np
import pandas as pd


def generate_strategy_signal(df_ml: pd.DataFrame) -> pd.Series:
    """
    Genera señal técnica simple:
    1 = estar invertido
    0 = estar fuera del mercado

    Reglas:
    - Precio sobre MA20
    - MA20 sobre MA50
    - RSI entre 40 y 70
    """

    signal = (
        (df_ml["precio"] > df_ml["ma20"]) &
        (df_ml["ma20"] > df_ml["ma50"]) &
        (df_ml["rsi"] >= 40) &
        (df_ml["rsi"] <= 70)
    ).astype(int)

    return signal


def run_backtest(
    df_ml: pd.DataFrame,
    initial_capital: float = 10_000,
    commission: float = 0.001,
    slippage: float = 0.0005
) -> pd.DataFrame:
    """
    Backtesting vectorizado con costos.

    Parámetros:
    - initial_capital: capital inicial simulado.
    - commission: comisión por operación. Ejemplo: 0.001 = 0.10%.
    - slippage: pérdida estimada por ejecución. Ejemplo: 0.0005 = 0.05%.

    Control de sesgo:
    - La señal se desplaza 1 período para evitar look-ahead bias.
    - La decisión de hoy se ejecuta usando información disponible hasta ayer.
    """

    df = df_ml.copy()

    df["asset_return"] = df["precio"].pct_change().fillna(0)

    raw_signal = generate_strategy_signal(df)

    # Evita look-ahead bias
    df["signal"] = raw_signal.shift(1).fillna(0)

    # Cambio de posición: compra o venta
    df["position_change"] = df["signal"].diff().abs().fillna(0)

    # Costo total por transacción
    transaction_cost = commission + slippage

    df["cost"] = df["position_change"] * transaction_cost

    # Retorno neto de estrategia
    df["strategy_return"] = (df["signal"] * df["asset_return"]) - df["cost"]

    # Curvas de capital
    df["buy_hold_equity"] = initial_capital * (1 + df["asset_return"]).cumprod()
    df["strategy_equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()

    # Drawdown
    df["strategy_peak"] = df["strategy_equity"].cummax()
    df["strategy_drawdown"] = (
        df["strategy_equity"] / df["strategy_peak"] - 1
    )

    df["buy_hold_peak"] = df["buy_hold_equity"].cummax()
    df["buy_hold_drawdown"] = (
        df["buy_hold_equity"] / df["buy_hold_peak"] - 1
    )

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_backtest_metrics(df_backtest: pd.DataFrame) -> dict:
    """
    Calcula métricas ejecutivas del backtest.
    """

    strategy_return = (
        df_backtest["strategy_equity"].iloc[-1] /
        df_backtest["strategy_equity"].iloc[0] - 1
    ) * 100

    buy_hold_return = (
        df_backtest["buy_hold_equity"].iloc[-1] /
        df_backtest["buy_hold_equity"].iloc[0] - 1
    ) * 100

    excess_return = strategy_return - buy_hold_return

    max_drawdown_strategy = df_backtest["strategy_drawdown"].min() * 100
    max_drawdown_buy_hold = df_backtest["buy_hold_drawdown"].min() * 100

    exposure = df_backtest["signal"].mean() * 100

    trades = int(df_backtest["position_change"].sum())

    turnover = df_backtest["position_change"].mean() * 100

    total_cost = df_backtest["cost"].sum() * 100

    daily_returns = df_backtest["strategy_return"]

    if daily_returns.std() != 0:
        sharpe_ratio = (
            daily_returns.mean() / daily_returns.std()
        ) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    return {
        "strategy_return": round(strategy_return, 2),
        "buy_hold_return": round(buy_hold_return, 2),
        "excess_return": round(excess_return, 2),
        "max_drawdown_strategy": round(max_drawdown_strategy, 2),
        "max_drawdown_buy_hold": round(max_drawdown_buy_hold, 2),
        "exposure": round(exposure, 2),
        "trades": trades,
        "turnover": round(turnover, 2),
        "total_cost": round(total_cost, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
    }
    
