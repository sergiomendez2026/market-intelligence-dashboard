# src/features.py

import pandas as pd


def create_ml_dataset(
    precio: pd.Series,
    ma20: pd.Series,
    ma50: pd.Series,
    rsi: pd.Series,
    std20: pd.Series
) -> pd.DataFrame:
    """
    Construye el dataset de Machine Learning para predicción de precio siguiente.
    """

    p = precio.values.astype(float)
    idx = precio.index

    ret1 = pd.Series(p, index=idx).pct_change(1)
    ret5 = pd.Series(p, index=idx).pct_change(5)

    df_ml = pd.DataFrame({
        "precio": p,
        "ma20": ma20.values.astype(float),
        "ma50": ma50.values.astype(float),
        "rsi": rsi.values.astype(float),
        "std20": std20.values.astype(float),
        "retorno_1d": ret1.values,
        "retorno_5d": ret5.values,
        "target": pd.Series(p, index=idx).shift(-1).values
    }, index=idx)

    return df_ml.dropna()


def get_feature_columns() -> list[str]:
    return [
        "precio",
        "ma20",
        "ma50",
        "rsi",
        "std20",
        "retorno_1d",
        "retorno_5d",
    ]
