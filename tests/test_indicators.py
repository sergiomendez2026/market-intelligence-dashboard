
import pandas as pd
import numpy as np

from src.indicators import add_rsi, add_moving_averages, add_bollinger_bands


def test_add_moving_averages():
    df = pd.DataFrame({
        "Close": np.arange(1, 101, dtype=float)
    })

    result = add_moving_averages(df)

    assert "MA20" in result.columns
    assert "MA50" in result.columns
    assert result["MA20"].notna().sum() > 0
    assert result["MA50"].notna().sum() > 0


def test_add_bollinger_bands():
    df = pd.DataFrame({
        "Close": np.arange(1, 101, dtype=float)
    })

    result = add_bollinger_bands(df)

    assert "BB_Upper" in result.columns
    assert "BB_Lower" in result.columns
    assert "BB_Middle" in result.columns


def test_add_rsi():
    df = pd.DataFrame({
        "Close": np.arange(1, 101, dtype=float)
    })

    result = add_rsi(df)

    assert "RSI" in result.columns
    assert result["RSI"].dropna().between(0, 100).all()
