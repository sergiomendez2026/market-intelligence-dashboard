# src/features.py

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    df["close_lag_1"] = df["Close"].shift(1)
    df["close_lag_2"] = df["Close"].shift(2)
    df["close_lag_5"] = df["Close"].shift(5)

    df["volume_change"] = df["Volume"].pct_change()

    df["target_price_next"] = df["Close"].shift(-1)
    df["target_direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()


def get_feature_columns() -> list[str]:
    return [
        "Close",
        "Volume",
        "MA20",
        "MA50",
        "BB_Width",
        "RSI",
        "return_1d",
        "return_5d",
        "volatility_10d",
        "volatility_20d",
        "close_lag_1",
        "close_lag_2",
        "close_lag_5",
        "volume_change",
    ]
