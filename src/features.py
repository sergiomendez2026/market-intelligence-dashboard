# src/features.py

import numpy as np
import pandas as pd


def create_ml_dataset(
    precio: pd.Series,
    ma20: pd.Series,
    ma50: pd.Series,
    rsi: pd.Series,
    std20: pd.Series
) -> pd.DataFrame:
    """
    Construye el dataset de Machine Learning para predicción del siguiente período.
    Incluye retornos, volatilidad, momentum y relaciones con medias móviles.
    """

    p = precio.astype(float)
    idx = precio.index

    df_ml = pd.DataFrame(index=idx)

    df_ml["precio"] = p
    df_ml["ma20"] = ma20.astype(float)
    df_ml["ma50"] = ma50.astype(float)
    df_ml["rsi"] = rsi.astype(float)
    df_ml["std20"] = std20.astype(float)

    # Retornos
    df_ml["retorno_1d"] = p.pct_change(1)
    df_ml["retorno_5d"] = p.pct_change(5)
    df_ml["retorno_10d"] = p.pct_change(10)
    df_ml["retorno_20d"] = p.pct_change(20)

    # Volatilidad de retornos
    df_ml["volatilidad_5d"] = df_ml["retorno_1d"].rolling(5).std()
    df_ml["volatilidad_10d"] = df_ml["retorno_1d"].rolling(10).std()
    df_ml["volatilidad_20d"] = df_ml["retorno_1d"].rolling(20).std()

    # Distancia relativa a medias móviles
    df_ml["distancia_ma20"] = (p - ma20) / ma20.replace(0, np.nan)
    df_ml["distancia_ma50"] = (p - ma50) / ma50.replace(0, np.nan)

    # Relación entre medias
    df_ml["ma20_vs_ma50"] = (ma20 - ma50) / ma50.replace(0, np.nan)

    # Momentum
    df_ml["momentum_10d"] = p - p.shift(10)
    df_ml["momentum_20d"] = p - p.shift(20)

    # Target de regresión
    df_ml["target"] = p.shift(-1)

    # Target de clasificación direccional
    df_ml["target_direction"] = (p.shift(-1) > p).astype(int)

    return df_ml.replace([np.inf, -np.inf], np.nan).dropna()


def get_feature_columns() -> list[str]:
    return [
        "precio",
        "ma20",
        "ma50",
        "rsi",
        "std20",
        "retorno_1d",
        "retorno_5d",
        "retorno_10d",
        "retorno_20d",
        "volatilidad_5d",
        "volatilidad_10d",
        "volatilidad_20d",
        "distancia_ma20",
        "distancia_ma50",
        "ma20_vs_ma50",
        "momentum_10d",
        "momentum_20d",
    ]
