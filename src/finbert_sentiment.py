# src/finbert_sentiment.py

from functools import lru_cache

import numpy as np
import pandas as pd


FINBERT_MODEL_NAME = "ProsusAI/finbert"


def normalize_finbert_label(label: str) -> str:
    """
    Normaliza etiquetas devueltas por modelos FinBERT.

    Etiquetas esperadas:
    - positive
    - negative
    - neutral
    """

    label = str(label).lower().strip()

    if "positive" in label:
        return "positive"

    if "negative" in label:
        return "negative"

    if "neutral" in label:
        return "neutral"

    return "neutral"


def sentiment_label_to_score(label: str, confidence: float) -> float:
    """
    Convierte etiqueta FinBERT en score numérico.

    positive -> +confidence
    negative -> -confidence
    neutral  -> 0
    """

    normalized = normalize_finbert_label(label)

    if normalized == "positive":
        return float(confidence)

    if normalized == "negative":
        return -float(confidence)

    return 0.0


@lru_cache(maxsize=1)
def load_finbert_pipeline(model_name: str = FINBERT_MODEL_NAME):
    """
    Carga el pipeline FinBERT.

    Se usa cache para evitar recargar el modelo repetidamente.
    """

    from transformers import pipeline

    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
    )


def classify_texts_with_finbert(
    texts: list[str],
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = 8,
    max_length: int = 256,
) -> pd.DataFrame:
    """
    Clasifica textos financieros usando FinBERT.

    Retorna un DataFrame con:
    - text
    - finbert_label
    - finbert_confidence
    - finbert_score
    """

    clean_texts = [
        str(text).strip()
        for text in texts
        if str(text).strip()
    ]

    if len(clean_texts) == 0:
        return pd.DataFrame(
            columns=[
                "text",
                "finbert_label",
                "finbert_confidence",
                "finbert_score",
            ]
        )

    classifier = load_finbert_pipeline(model_name)

    results = classifier(
        clean_texts,
        batch_size=batch_size,
        max_length=max_length,
        truncation=True,
    )

    records = []

    for text, result in zip(clean_texts, results):
        label = normalize_finbert_label(result.get("label", "neutral"))
        confidence = float(result.get("score", 0.0))
        score = sentiment_label_to_score(label, confidence)

        records.append(
            {
                "text": text,
                "finbert_label": label,
                "finbert_confidence": confidence,
                "finbert_score": score,
            }
        )

    return pd.DataFrame(records)


def prepare_news_dataframe(
    news_df: pd.DataFrame,
    date_column: str,
    text_column: str,
) -> pd.DataFrame:
    """
    Limpia un DataFrame de noticias.

    Requiere:
    - columna de fecha
    - columna de texto/titular/noticia
    """

    if date_column not in news_df.columns:
        raise ValueError(f"No existe la columna de fecha: {date_column}")

    if text_column not in news_df.columns:
        raise ValueError(f"No existe la columna de texto: {text_column}")

    prepared = news_df[[date_column, text_column]].copy()
    prepared = prepared.rename(
        columns={
            date_column: "date",
            text_column: "text",
        }
    )

    prepared["date"] = pd.to_datetime(
        prepared["date"],
        errors="coerce",
    ).dt.normalize()

    prepared["text"] = prepared["text"].astype(str).str.strip()

    prepared = prepared.dropna(subset=["date", "text"])
    prepared = prepared[prepared["text"] != ""]

    return prepared.reset_index(drop=True)


def build_finbert_classified_news(
    news_df: pd.DataFrame,
    date_column: str,
    text_column: str,
    model_name: str = FINBERT_MODEL_NAME,
    max_articles: int = 500,
) -> pd.DataFrame:
    """
    Clasifica noticias con FinBERT manteniendo la fecha original.

    max_articles limita el costo computacional en Streamlit Cloud.
    """

    prepared = prepare_news_dataframe(
        news_df=news_df,
        date_column=date_column,
        text_column=text_column,
    )

    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "text",
                "finbert_label",
                "finbert_confidence",
                "finbert_score",
            ]
        )

    prepared = prepared.sort_values("date").tail(max_articles).reset_index(drop=True)

    classified = classify_texts_with_finbert(
        texts=prepared["text"].tolist(),
        model_name=model_name,
    )

    result = pd.concat(
        [
            prepared[["date", "text"]].reset_index(drop=True),
            classified[
                [
                    "finbert_label",
                    "finbert_confidence",
                    "finbert_score",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    return result


def aggregate_classified_sentiment_by_date(
    classified_news: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega sentimiento FinBERT por fecha.

    Genera variables diarias:
    - sentiment_score_mean
    - sentiment_score_sum
    - news_count
    - positive_share
    - negative_share
    - neutral_share
    - confidence_mean
    """

    required_columns = [
        "date",
        "finbert_label",
        "finbert_confidence",
        "finbert_score",
    ]

    missing = [
        col for col in required_columns
        if col not in classified_news.columns
    ]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = classified_news.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    if df.empty:
        return pd.DataFrame()

    df["is_positive"] = (df["finbert_label"] == "positive").astype(int)
    df["is_negative"] = (df["finbert_label"] == "negative").astype(int)
    df["is_neutral"] = (df["finbert_label"] == "neutral").astype(int)

    grouped = df.groupby("date").agg(
        sentiment_score_mean=("finbert_score", "mean"),
        sentiment_score_sum=("finbert_score", "sum"),
        news_count=("finbert_score", "count"),
        positive_share=("is_positive", "mean"),
        negative_share=("is_negative", "mean"),
        neutral_share=("is_neutral", "mean"),
        confidence_mean=("finbert_confidence", "mean"),
    )

    return grouped


def build_daily_finbert_features(
    classified_news: pd.DataFrame,
    price_index: pd.Index,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Construye variables diarias de sentimiento alineadas al índice de precios.

    Se aplica lag_days=1 por defecto para reducir look-ahead bias:
    el sentimiento observado en t se usa para predecir t+1.
    """

    price_dates = pd.to_datetime(price_index).normalize()

    daily_sentiment = aggregate_classified_sentiment_by_date(classified_news)

    if daily_sentiment.empty:
        features = pd.DataFrame(index=price_index)
        features["finbert_score_mean"] = 0.0
        features["finbert_score_sum"] = 0.0
        features["finbert_news_count"] = 0
        features["finbert_positive_share"] = 0.0
        features["finbert_negative_share"] = 0.0
        features["finbert_neutral_share"] = 1.0
        features["finbert_confidence_mean"] = 0.0
        return features

    full = pd.DataFrame(index=pd.Index(price_dates, name="date"))
    full = full.join(daily_sentiment, how="left")

    full = full.fillna(
        {
            "sentiment_score_mean": 0.0,
            "sentiment_score_sum": 0.0,
            "news_count": 0,
            "positive_share": 0.0,
            "negative_share": 0.0,
            "neutral_share": 1.0,
            "confidence_mean": 0.0,
        }
    )

    if lag_days > 0:
        full = full.shift(lag_days)

    full = full.fillna(
        {
            "sentiment_score_mean": 0.0,
            "sentiment_score_sum": 0.0,
            "news_count": 0,
            "positive_share": 0.0,
            "negative_share": 0.0,
            "neutral_share": 1.0,
            "confidence_mean": 0.0,
        }
    )

    features = pd.DataFrame(index=price_index)
    features["finbert_score_mean"] = full["sentiment_score_mean"].values
    features["finbert_score_sum"] = full["sentiment_score_sum"].values
    features["finbert_news_count"] = full["news_count"].values
    features["finbert_positive_share"] = full["positive_share"].values
    features["finbert_negative_share"] = full["negative_share"].values
    features["finbert_neutral_share"] = full["neutral_share"].values
    features["finbert_confidence_mean"] = full["confidence_mean"].values

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_finbert_sentiment_features_from_news(
    news_df: pd.DataFrame,
    price_index: pd.Index,
    date_column: str,
    text_column: str,
    model_name: str = FINBERT_MODEL_NAME,
    lag_days: int = 1,
    max_articles: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo:

    1. Recibe noticias.
    2. Clasifica con FinBERT.
    3. Agrega sentimiento por fecha.
    4. Genera features alineadas a precios.
    """

    classified_news = build_finbert_classified_news(
        news_df=news_df,
        date_column=date_column,
        text_column=text_column,
        model_name=model_name,
        max_articles=max_articles,
    )

    sentiment_features = build_daily_finbert_features(
        classified_news=classified_news,
        price_index=price_index,
        lag_days=lag_days,
    )

    return sentiment_features, classified_news
