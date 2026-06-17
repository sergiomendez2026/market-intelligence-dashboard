# src/backtesting.py

import numpy as np
import pandas as pd


def run_backtest(
    df_ml: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    slippage: float = 0.001
) -> pd.DataFrame:
    """
    Backtesting vectorizado con costos transaccionales.

    Supuestos:
    - Señal técnica basada en tendencia y momentum.
    - La señal se desplaza un período para reducir look-ahead bias.
    - Se incluyen comisión y slippage por cambio de posición.
    - Posición:
        1 = dentro del mercado
        0 = fuera del mercado
    """

    df = df_ml.copy()

    if "precio" not in df.columns:
        raise ValueError("df_ml debe contener la columna 'precio'.")

    required_columns = ["ma20", "ma50", "rsi"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas para backtesting: {missing_columns}")

    df["market_return"] = df["precio"].pct_change()

    # =========================
    # Reglas de señal
    # =========================
    # Entrada cuando:
    # precio > MA20
    # MA20 > MA50
    # RSI no está en sobrecompra extrema
    df["raw_signal"] = np.where(
        (df["precio"] > df["ma20"]) &
        (df["ma20"] > df["ma50"]) &
        (df["rsi"] < 70),
        1,
        0
    )

    # Desplazamiento para evitar usar información del mismo período
    df["position"] = df["raw_signal"].shift(1).fillna(0)

    # Cambio de posición: entrada o salida
    df["trade"] = df["position"].diff().abs().fillna(0)

    transaction_cost = commission + slippage

    df["gross_strategy_return"] = df["position"] * df["market_return"]

    df["transaction_cost"] = df["trade"] * transaction_cost

    df["net_strategy_return"] = (
        df["gross_strategy_return"] - df["transaction_cost"]
    )

    df["buy_hold_equity"] = (
        initial_capital * (1 + df["market_return"].fillna(0)).cumprod()
    )

    df["strategy_equity"] = (
        initial_capital * (1 + df["net_strategy_return"].fillna(0)).cumprod()
    )

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
    Calcula métricas principales del backtest.
    """

    if df_backtest.empty:
        return {
            "strategy_return": 0.0,
            "buy_hold_return": 0.0,
            "excess_return": 0.0,
            "max_drawdown_strategy": 0.0,
            "max_drawdown_buy_hold": 0.0,
            "exposure": 0.0,
            "number_of_trades": 0,
            "turnover": 0.0,
            "total_transaction_cost": 0.0,
            "sharpe_ratio": 0.0,
        }

    strategy_initial = df_backtest["strategy_equity"].iloc[0]
    strategy_final = df_backtest["strategy_equity"].iloc[-1]

    buy_hold_initial = df_backtest["buy_hold_equity"].iloc[0]
    buy_hold_final = df_backtest["buy_hold_equity"].iloc[-1]

    strategy_return = (strategy_final / strategy_initial - 1) * 100
    buy_hold_return = (buy_hold_final / buy_hold_initial - 1) * 100

    excess_return = strategy_return - buy_hold_return

    max_drawdown_strategy = df_backtest["strategy_drawdown"].min() * 100
    max_drawdown_buy_hold = df_backtest["buy_hold_drawdown"].min() * 100

    exposure = df_backtest["position"].mean() * 100

    number_of_trades = int(df_backtest["trade"].sum())

    turnover = df_backtest["trade"].mean() * 100

    total_transaction_cost = df_backtest["transaction_cost"].sum() * 100

    daily_returns = df_backtest["net_strategy_return"].dropna()

    if daily_returns.std() == 0 or len(daily_returns) < 2:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (
            daily_returns.mean() / daily_returns.std()
        ) * np.sqrt(252)

    return {
        "strategy_return": round(float(strategy_return), 2),
        "buy_hold_return": round(float(buy_hold_return), 2),
        "excess_return": round(float(excess_return), 2),
        "max_drawdown_strategy": round(float(max_drawdown_strategy), 2),
        "max_drawdown_buy_hold": round(float(max_drawdown_buy_hold), 2),
        "exposure": round(float(exposure), 2),
        "number_of_trades": number_of_trades,
        "turnover": round(float(turnover), 2),
        "total_transaction_cost": round(float(total_transaction_cost), 2),
        "sharpe_ratio": round(float(sharpe_ratio), 2),
    }
    
