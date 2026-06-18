# tests/test_explainability.py

import numpy as np
import pandas as pd

from src.explainability import (
    calculate_random_forest_feature_importance,
    classify_feature_group,
    summarize_feature_importance_by_group,
    get_top_features,
)


def test_calculate_random_forest_feature_importance():
    np.random.seed(42)

    X = pd.DataFrame(
        {
            "return_1": np.random.normal(0, 1, 120),
            "rsi_14": np.random.normal(50, 10, 120),
            "volatility_20": np.random.uniform(0, 1, 120),
            "finbert_score_mean": np.random.normal(0, 1, 120),
        }
    )

    y = pd.Series(np.random.randint(0, 2, 120))

    result = calculate_random_forest_feature_importance(X, y)

    assert not result.empty
    assert "Variable" in result.columns
    assert "Importancia" in result.columns
    assert "Importancia (%)" in result.columns
    assert len(result) == X.shape[1]


def test_classify_feature_group():
    assert classify_feature_group("finbert_score_mean") == "Sentimiento FinBERT"
    assert classify_feature_group("rsi_14") == "Momentum / RSI"
    assert classify_feature_group("volatility_20") == "Volatilidad"
    assert classify_feature_group("return_1") == "Retornos / Momentum"


def test_summarize_feature_importance_by_group():
    importance_df = pd.DataFrame(
        {
            "Variable": ["return_1", "rsi_14", "finbert_score_mean"],
            "Importancia": [0.3, 0.2, 0.5],
            "Importancia (%)": [30.0, 20.0, 50.0],
        }
    )

    summary = summarize_feature_importance_by_group(importance_df)

    assert not summary.empty
    assert "Grupo" in summary.columns
    assert "Importancia (%)" in summary.columns


def test_get_top_features():
    importance_df = pd.DataFrame(
        {
            "Variable": ["a", "b", "c"],
            "Importancia": [0.5, 0.3, 0.2],
            "Importancia (%)": [50.0, 30.0, 20.0],
        }
    )

    top = get_top_features(importance_df, top_n=2)

    assert len(top) == 2
