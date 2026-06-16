# src/backtesting.py

import numpy as np
import pandas as pd


def generate_signal_from_features(df_ml: pd.DataFrame) -> pd.Series:
    """
    Genera una señal simple basada en condiciones técnicas.
    1 = posición larga
    0 = fuera del mercado

    Esta es una señal base para backtesting.
    Luego será reemplazada por market_signal_score.
    """

    signal = (
        (df_ml["precio"] > df_ml["ma20"]) &
        (df_ml["ma20"] > df_ml["ma50"]) &
        (df_ml["rsi"] > 40) &
        (df_ml["rsi"] < 70)
    ).astype(int)

    return signal


def run_backtest(
    df_ml: pd.DataFrame,
    initial_capital: float = 10_000.0
) -> pd.DataFrame:
    """
    Backtesting vectorizado simple.

    Estrategia:
    - Si signal = 1, toma exposición al activo.
    - Si signal = 0, queda en efectivo.
    - La señal se desplaza un período para evitar look-ahead bias.
    """

    df_bt = df_ml.copy()

    df_bt["market_return"] = df_bt["precio"].pct_change()

    df_bt["signal"] = generate_signal_from_features(df_bt)

    # Evita look-ahead bias: uso la señal de ayer para operar hoy
    df_bt["strategy_position"] = df_bt["signal"].shift(1).fillna(0)

    df_bt["strategy_return"] = (
        df_bt["strategy_position"] * df_bt["market_return"]
    )

    df_bt["buy_hold_equity"] = (
        initial_capital * (1 + df_bt["market_return"].fillna(0)).cumprod()
    )

    df_bt["strategy_equity"] = (
        initial_capital * (1 + df_bt["strategy_return"].fillna(0)).cumprod()
    )

    return df_bt.dropna()


def calculate_backtest_metrics(df_bt: pd.DataFrame) -> dict:
    """
    Calcula métricas ejecutivas del backtesting.
    """

    if df_bt.empty:
        return {
            "strategy_return": 0.0,
            "buy_hold_return": 0.0,
            "excess_return": 0.0,
            "max_drawdown_strategy": 0.0,
            "exposure": 0.0,
        }

    initial_strategy = df_bt["strategy_equity"].iloc[0]
    final_strategy = df_bt["strategy_equity"].iloc[-1]

    initial_bh = df_bt["buy_hold_equity"].iloc[0]
    final_bh = df_bt["buy_hold_equity"].iloc[-1]

    strategy_return = ((final_strategy / initial_strategy) - 1) * 100
    buy_hold_return = ((final_bh / initial_bh) - 1) * 100

    strategy_peak = df_bt["strategy_equity"].cummax()
    strategy_drawdown = (df_bt["strategy_equity"] / strategy_peak) - 1
    max_drawdown_strategy = strategy_drawdown.min() * 100

    exposure = df_bt["strategy_position"].mean() * 100

    return {
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": strategy_return - buy_hold_return,
        "max_drawdown_strategy": max_drawdown_strategy,
        "exposure": exposure,
    }
