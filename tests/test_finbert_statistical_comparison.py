# tests/test_finbert_statistical_comparison.py

import numpy as np
import pandas as pd

from src.statistical_tests import compare_technical_vs_sentiment_statistically
from src.walkforward_academic import collect_walkforward_predictions_technical_vs_sentiment


def test_compare_technical_vs_sentiment_statistically():
    df = pd.DataFrame(
        {
            "actual_return": [
                0.01,
                -0.02,
                0.03,
                -0.01,
                0.02,
                -0.03,
                0.01,
                -0.01,
                0.02,
                -0.02,
            ],
            "actual_direction": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "technical_pred_return": [
                0.01,
                0.01,
                0.02,
                -0.02,
                -0.01,
                -0.01,
                0.02,
                0.01,
                -0.01,
                -0.02,
            ],
            "technical_pred_direction": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            "sentiment_pred_return": [
                0.01,
                -0.01,
                0.02,
                -0.02,
                0.01,
                -0.01,
                0.02,
                -0.01,
                0.01,
                -0.02,
            ],
            "sentiment_pred_direction": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    results = compare_technical_vs_sentiment_statistically(df)

    assert not results.empty
    assert "Comparación" in results.columns
    assert "Test" in results.columns


def test_collect_walkforward_predictions_technical_vs_sentiment():
    prices = pd.Series(
        np.linspace(100, 150, 320)
        + np.random.normal(0, 1, 320)
    )

    sentiment_features = pd.DataFrame(
        {
            "finbert_score_mean": np.random.normal(0, 0.1, 320),
            "finbert_score_sum": np.random.normal(0, 0.2, 320),
            "finbert_news_count": np.random.randint(0, 5, 320),
            "finbert_positive_share": np.random.uniform(0, 1, 320),
            "finbert_negative_share": np.random.uniform(0, 1, 320),
            "finbert_neutral_share": np.random.uniform(0, 1, 320),
            "finbert_confidence_mean": np.random.uniform(0, 1, 320),
        },
        index=prices.index,
    )

    result = collect_walkforward_predictions_technical_vs_sentiment(
        prices=prices,
        sentiment_features=sentiment_features,
        initial_train_size=120,
        test_window=20,
        step_size=20,
        max_windows=2,
    )

    assert not result.empty
    assert "technical_pred_direction" in result.columns
    assert "sentiment_pred_direction" in result.columns
    assert "actual_direction" in result.columns
