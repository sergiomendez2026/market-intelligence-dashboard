# src/sentiment.py

import re
import numpy as np


POSITIVE_WORDS = [
    "beat", "growth", "profit", "profits", "bullish", "upgrade",
    "strong", "record", "surge", "rally", "gain", "gains",
    "outperform", "positive", "optimistic", "buy"
]

NEGATIVE_WORDS = [
    "miss", "loss", "losses", "bearish", "downgrade",
    "weak", "drop", "decline", "fall", "risk", "lawsuit",
    "recession", "negative", "sell", "cut", "cuts"
]


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_fallback_sentiment_score(texts: list[str] | None = None) -> dict:
    """
    Sentimiento financiero liviano basado en diccionario.
    Devuelve score 0-100.
    50 = neutral.
    """

    if not texts:
        return {
            "sentiment_score": 50.0,
            "sentiment_label": "Neutral",
            "positive_hits": 0,
            "negative_hits": 0,
            "method": "Fallback neutral"
        }

    joined_text = " ".join([clean_text(text) for text in texts])

    positive_hits = sum(
        len(re.findall(rf"\b{word}\b", joined_text))
        for word in POSITIVE_WORDS
    )

    negative_hits = sum(
        len(re.findall(rf"\b{word}\b", joined_text))
        for word in NEGATIVE_WORDS
    )

    total_hits = positive_hits + negative_hits

    if total_hits == 0:
        score = 50.0
    else:
        raw_score = (positive_hits - negative_hits) / total_hits
        score = 50 + raw_score * 50

    score = float(np.clip(score, 0, 100))

    if score >= 60:
        label = "Positivo"
    elif score <= 40:
        label = "Negativo"
    else:
        label = "Neutral"

    return {
        "sentiment_score": round(score, 2),
        "sentiment_label": label,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "method": "Keyword fallback"
    }
