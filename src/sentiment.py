# src/sentiment.py

from transformers import pipeline


def load_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )


def analyze_headlines(headlines: list[str]) -> dict:
    analyzer = load_sentiment_analyzer()
    results = analyzer(headlines)

    score_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1,
    }

    sentiment_scores = []
    confidence_scores = []

    for result in results:
        label = result["label"].lower()
        score = result["score"]

        sentiment_scores.append(score_map.get(label, 0))
        confidence_scores.append(score)

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    return {
        "avg_sentiment": avg_sentiment,
        "avg_confidence": avg_confidence,
        "positive_count": sentiment_scores.count(1),
        "neutral_count": sentiment_scores.count(0),
        "negative_count": sentiment_scores.count(-1),
    }
