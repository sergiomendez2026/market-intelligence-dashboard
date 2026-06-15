# src/indicators.py

import pandas as pd


def add_moving_averages(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=short_window).mean()
    df["MA50"] = df["Close"].rolling(window=long_window).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()

    df["BB_Middle"] = rolling_mean
    df["BB_Upper"] = rolling_mean + num_std * rolling_std
    df["BB_Lower"] = rolling_mean - num_std * rolling_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_moving_averages(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    return df.dropna()
