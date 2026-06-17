# tests/test_walkforward_academic.py

import numpy as np
import pandas as pd

from src.walkforward_academic import (
    build_basic_technical_features,
    build_supervised_financial_dataset,
    compare_walkforward_academic_models,
)


def test_build_basic_technical_features():
    prices = pd.Series(
        np.linspace(100, 150, 300)
        + np.random.normal(0, 1, 300)
    )

    features = build_basic_technical_features(prices)

    assert not features.empty
    assert "return_1" in features.columns
    assert "rsi_14" in features.columns
    assert "volatility_20" in features.columns


def test_build_supervised_financial_dataset():
    prices = pd.Series(
        np.linspace(100, 150, 300)
        + np.random.normal(0, 1, 300)
    )

    dataset = build_supervised_financial_dataset(prices)

    assert not dataset.empty
    assert "future_return" in dataset.columns
    assert "future_direction" in dataset.columns
    assert "close" in dataset.columns


def test_compare_walkforward_academic_models_without_arima():
    prices = pd.Series(
        np.linspace(100, 150, 320)
        + np.random.normal(0, 1, 320)
    )

    results = compare_walkforward_academic_models(
        prices=prices,
        initial_train_size=120,
        test_window=20,
        step_size=20,
        max_windows=2,
        include_arima=False,
    )

    assert not results.empty
    assert "Modelo" in results.columns
    assert "F1 Score" in results.columns
  
