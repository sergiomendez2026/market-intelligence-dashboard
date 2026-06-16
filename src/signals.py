# src/signals.py

import numpy as np


def calculate_model_probability_score(model_probability=None) -> float:
    """
    Convierte la probabilidad alcista del modelo direccional a escala 0-100.
    """

    if model_probability is None:
        return 50.0

    try:
        if np.isnan(model_probability):
            return 50.0
    except TypeError:
        return 50.0

    return float(np.clip(model_probability * 100, 0, 100))


def calculate_technical_score(
    last_price: float,
    rsi: float,
    ma20: float,
    ma50: float
) -> float:
    """
    Score técnico basado en tendencia y momentum.
    """

    score = 50.0

    if last_price > ma20:
        score += 12.5
    else:
        score -= 12.5

    if last_price > ma50:
        score += 12.5
    else:
        score -= 12.5

    if ma20 > ma50:
        score += 12.5
    else:
        score -= 12.5

    if 45 <= rsi <= 65:
        score += 12.5
    elif 30 <= rsi < 45:
        score += 5.0
    elif 65 < rsi <= 70:
        score += 5.0
    elif rsi < 30:
        score -= 7.5
    elif rsi > 70:
        score -= 10.0

    return float(np.clip(score, 0, 100))


def calculate_volatility_score(volatility: float) -> float:
    """
    Score de volatilidad.
    Menor volatilidad = mayor score.
    """

    if volatility is None:
        return 50.0

    try:
        if np.isnan(volatility):
            return 50.0
    except TypeError:
        return 50.0

    volatility_score = 100 - (volatility / 0.05) * 100

    return float(np.clip(volatility_score, 0, 100))


def calculate_sentiment_score(sentiment_score=None) -> float:
    """
    Score de sentimiento.
    Por ahora usamos 50 como neutral si no hay FinBERT.
    """

    if sentiment_score is None:
        return 50.0

    try:
        if np.isnan(sentiment_score):
            return 50.0
    except TypeError:
        return 50.0

    return float(np.clip(sentiment_score, 0, 100))


def classify_market_signal(score: float) -> tuple[str, str]:
    """
    Clasifica el Market Signal Score.
    """

    if score >= 80:
        return (
            "Strong Bullish",
            "El activo muestra una señal alcista fuerte según el modelo integrado."
        )

    if score >= 60:
        return (
            "Bullish moderado",
            "El activo muestra fortaleza moderada según las señales actuales."
        )

    if score >= 40:
        return (
            "Neutral",
            "El activo muestra señales mixtas o sin dirección dominante."
        )

    if score >= 20:
        return (
            "Bearish moderado",
            "El activo muestra debilidad moderada según las señales actuales."
        )

    return (
        "Strong Bearish",
        "El activo muestra una señal bajista fuerte según el modelo integrado."
    )


def compute_market_signal(
    last_price: float,
    rsi: float,
    ma20: float,
    ma50: float,
    volatility: float,
    model_probability=None,
    sentiment_score=None,
    predicted_price=None
) -> dict:
    """
    Market Signal Score integrado.

    Fórmula:
    40% probabilidad alcista del modelo direccional
    25% score técnico
    20% sentimiento financiero
    15% ajuste por volatilidad

    predicted_price queda como argumento opcional para compatibilidad,
    pero ya no se usa en la fórmula principal.
    """

    model_probability_score = calculate_model_probability_score(
        model_probability
    )

    technical_score = calculate_technical_score(
        last_price=last_price,
        rsi=rsi,
        ma20=ma20,
        ma50=ma50
    )

    sentiment_score_value = calculate_sentiment_score(
        sentiment_score
    )

    volatility_score = calculate_volatility_score(
        volatility
    )

    market_signal_score = (
        0.40 * model_probability_score +
        0.25 * technical_score +
        0.20 * sentiment_score_value +
        0.15 * volatility_score
    )

    market_signal_score = round(float(np.clip(market_signal_score, 0, 100)), 2)

    signal, interpretation = classify_market_signal(market_signal_score)

    return {
        "market_signal_score": market_signal_score,
        "signal": signal,
        "interpretation": interpretation,
        "model_probability_score": round(model_probability_score, 2),
        "technical_score": round(technical_score, 2),
        "sentiment_score": round(sentiment_score_value, 2),
        "volatility_score": round(volatility_score, 2),
    }
