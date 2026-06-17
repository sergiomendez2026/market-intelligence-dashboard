# src/backtesting.py

import numpy as np
import pandas as pd


def generate_strategy_signal(df_ml: pd.DataFrame) -> pd.Series:
    """
    Genera una señal técnica simple:
    1 = estar comprado
    0 = estar fuera del mercado

    Regla:
    - Compra si precio > MA20 y RSI entre 40 y 70.
    - Sale si no cumple condiciones.
    """

    signal = (
        (df_ml["precio"] > df_ml["ma20"]) &
        (df_ml["rsi"] >= 40) &
        (df_ml["rsi"] <= 70)
    ).astype(int)

    return signal


def run_backtest(
    df_ml: pd.DataFrame,
    initial_capital: float = 10_000,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001
) -> pd.DataFrame:
    """
    Backtesting vectorizado con costos.

    commission_rate:
        Comisión por operación. Ejemplo: 0.001 = 0.10%

    slippage_rate:
        Deslizamiento estimado por operación. Ejemplo: 0.001 = 0.10%

    Nota:
    La señal se desplaza 1 período para reducir look-ahead bias.
    """

    df = df_ml.copy()

    df["market_return"] = df["precio"].pct_change().fillna(0)

    df["raw_signal"] = generate_strategy_signal(df)

    # Evita look-ahead bias: la señal de hoy se ejecuta al siguiente período
    df["position"] = df["raw_signal"].shift(1).fillna(0)

    # Detecta cambios de posición: entrada o salida
    df["trade"] = df["position"].diff().abs().fillna(0)

    transaction_cost = commission_rate + slippage_rate

    df["strategy_return_gross"] = df["position"] * df["market_return"]

    df["transaction_cost"] = df["trade"] * transaction_cost

    df["strategy_return_net"] = (
        df["strategy_return_gross"] - df["transaction_cost"]
    )

    df["buy_hold_equity"] = (
        initial_capital * (1 + df["market_return"]).cumprod()
    )

    df["strategy_equity"] = (
        initial_capital * (1 + df["strategy_return_net"]).cumprod()
    )

    df["running_max_strategy"] = df["strategy_equity"].cummax()

    df["drawdown_strategy"] = (
        df["strategy_equity"] / df["running_max_strategy"] - 1
    )

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_backtest_metrics(df_backtest: pd.DataFrame) -> dict:
    """
    Calcula métricas principales del backtesting.
    """

    if df_backtest.empty:
        return {
            "strategy_return": 0.0,
            "buy_hold_return": 0.0,
            "excess_return": 0.0,
            "max_drawdown_strategy": 0.0,
            "exposure": 0.0,
            "number_of_trades": 0,
            "total_transaction_cost": 0.0,
        }

    strategy_return = (
        df_backtest["strategy_equity"].iloc[-1]
        / df_backtest["strategy_equity"].iloc[0]
        - 1
    ) * 100

    buy_hold_return = (
        df_backtest["buy_hold_equity"].iloc[-1]
        / df_backtest["buy_hold_equity"].iloc[0]
        - 1
    ) * 100

    excess_return = strategy_return - buy_hold_return

    max_drawdown_strategy = df_backtest["drawdown_strategy"].min() * 100

    exposure = df_backtest["position"].mean() * 100

    number_of_trades = int(df_backtest["trade"].sum())

    total_transaction_cost = df_backtest["transaction_cost"].sum() * 100

    return {
        "strategy_return": round(strategy_return, 2),
        "buy_hold_return": round(buy_hold_return, 2),
        "excess_return": round(excess_return, 2),
        "max_drawdown_strategy": round(max_drawdown_strategy, 2),
        "exposure": round(exposure, 2),
        "number_of_trades": number_of_trades,
        "total_transaction_cost": round(total_transaction_cost, 2),
    }
    
