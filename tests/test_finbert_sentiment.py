# tests/test_finbert_sentiment.py

import pandas as pd

from src.finbert_sentiment import (
    aggregate_classified_sentiment_by_date,
    build_daily_finbert_features,
    normalize_finbert_label,
    sentiment_label_to_score,
)


def test_normalize_finbert_label():
    assert normalize_finbert_label("positive") == "positive"
    assert normalize_finbert_label("NEGATIVE") == "negative"
    assert normalize_finbert_label("Neutral") == "neutral"
    assert normalize_finbert_label("unknown") == "neutral"


def test_sentiment_label_to_score():
    assert sentiment_label_to_score("positive", 0.8) == 0.8
    assert sentiment_label_to_score("negative", 0.7) == -0.7
    assert sentiment_label_to_score("neutral", 0.9) == 0.0


def test_aggregate_classified_sentiment_by_date():
    classified_news = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
            ],
            "finbert_label": [
                "positive",
                "negative",
                "neutral",
            ],
            "finbert_confidence": [
                0.9,
                0.8,
                0.7,
            ],
            "finbert_score": [
                0.9,
                -0.8,
                0.0,
            ],
        }
    )

    daily = aggregate_classified_sentiment_by_date(classified_news)

    assert not daily.empty
    assert "sentiment_score_mean" in daily.columns
    assert "news_count" in daily.columns
    assert daily.loc[pd.Timestamp("2026-01-01"), "news_count"] == 2


def test_build_daily_finbert_features():
    classified_news = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-02",
            ],
            "finbert_label": [
                "positive",
                "negative",
            ],
            "finbert_confidence": [
                0.9,
                0.8,
            ],
            "finbert_score": [
                0.9,
                -0.8,
            ],
        }
    )

    price_index = pd.date_range("2026-01-01", periods=5, freq="D")

    features = build_daily_finbert_features(
        classified_news=classified_news,
        price_index=price_index,
        lag_days=1,
    )

    assert not features.empty
    assert "finbert_score_mean" in features.columns
    assert "finbert_news_count" in features.columns
    assert len(features) == len(price_index)
  
