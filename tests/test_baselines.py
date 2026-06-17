# tests/test_baselines.py

import numpy as np
import pandas as pd

from src.baselines import (
    calculate_direction,
    calculate_directional_metrics,
    calculate_regression_metrics,
    run_naive_price_baseline,
)


def test_calculate_direction():
    values = np.array([-0.1, 0.0, 0.2, 1.5, -2.0])
    result = calculate_direction(values)

    expected = np.array([0, 0, 1, 1, 0])

    assert np.array_equal(result, expected)


def test_calculate_regression_metrics():
    y_true = np.array([100, 110, 120])
    y_pred = np.array([98, 111, 119])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] > 0


def test_calculate_directional_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    metrics = calculate_directional_metrics(y_true, y_pred)

    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1 Score" in metrics
    assert metrics["Accuracy"] == 75.0


def test_run_naive_price_baseline():
    prices = pd.Series([100, 102, 101, 105, 107, 106])

    result = run_naive_price_baseline(prices)

    assert result.model_name == "Naive t+1 = precio actual"
    assert len(result.y_true) == len(result.y_pred)
    assert "MAE" in result.metrics
    assert "Accuracy" in result.metrics
