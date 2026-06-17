# tests/test_statistical_tests.py

import numpy as np
import pandas as pd

from src.statistical_tests import (
    bootstrap_metric_difference,
    compare_model_predictions_statistically,
    diebold_mariano_test,
    mcnemar_test,
)


def test_diebold_mariano_test():
    errors_a = np.array([0.10, 0.20, 0.15, 0.30, 0.25, 0.18, 0.12, 0.22, 0.19, 0.16])
    errors_b = np.array([0.08, 0.18, 0.12, 0.25, 0.21, 0.15, 0.10, 0.20, 0.16, 0.14])

    result = diebold_mariano_test(errors_a, errors_b)

    assert "p-value" in result
    assert "Estadístico" in result
    assert result["Test"] == "Diebold-Mariano"


def test_mcnemar_test():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    pred_a = np.array([1, 0, 0, 1, 0, 0])
    pred_b = np.array([1, 0, 1, 1, 0, 1])

    result = mcnemar_test(y_true, pred_a, pred_b)

    assert "p-value" in result
    assert "Estadístico" in result
    assert result["Test"] == "McNemar"


def test_bootstrap_metric_difference():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    pred_a = np.array([1, 0, 0, 1, 0, 0, 1, 0])
    pred_b = np.array([1, 0, 1, 1, 0, 1, 0, 0])

    result = bootstrap_metric_difference(
        y_true,
        pred_a,
        pred_b,
        metric="f1",
        n_bootstrap=100,
    )

    assert "Diferencia media B-A" in result
    assert "IC 95% inferior" in result
    assert "IC 95% superior" in result


def test_compare_model_predictions_statistically():
    df = pd.DataFrame(
        {
            "actual_return": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.01, 0.02, -0.02],
            "actual_direction": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "naive_pred_return": [0.0] * 10,
            "naive_pred_direction": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            "technical_pred_return": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01, -0.02],
            "technical_pred_direction": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    results = compare_model_predictions_statistically(df)

    assert not results.empty
    assert "Comparación" in results.columns
    assert "Test" in results.columns
