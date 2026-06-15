# src/indicators.py

import numpy as np
import pandas as pd


def validate_price_dataframe(df: pd.DataFrame) -> None:
    """
    Valida que el DataFrame tenga la estructura mínima requerida.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    if df["Close"].isna().all():
        raise ValueError("'Close' column cannot be fully empty.")


def add_moving_averages(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    """
    Agrega medias móviles simples al DataFrame.
    """
    validate_price_dataframe(df)

    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=short_window).mean()
    df["MA50"] = df["Close"].rolling(window=long_window).mean()

    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Agrega Bandas de Bollinger y ancho relativo de bandas.
    """
    validate_price_dataframe(df)

    df = df.copy()

    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()

    df["BB_Middle"] = rolling_mean
    df["BB_Upper"] = rolling_mean + num_std * rolling_std
    df["BB_Lower"] = rolling_mean - num_std * rolling_std

    denominator = df["BB_Middle"].replace(0, np.nan)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / denominator

    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Agrega RSI usando media móvil simple de ganancias y pérdidas.
    """
    validate_price_dataframe(df)

    df = df.copy()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # Valor neutral cuando no hay suficiente información o la división no es estable
    df["RSI"] = df["RSI"].replace([np.inf, -np.inf], np.nan)
    df["RSI"] = df["RSI"].fillna(50)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega todos los indicadores técnicos usados por el dashboard.
    """
    validate_price_dataframe(df)

    df = df.copy()
    df = add_moving_averages(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)

    return df.dropna()
