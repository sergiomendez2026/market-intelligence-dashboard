# src/signals.py

import numpy as np


def calculate_model_probability_score(model_probability: float) -> float:
    """
    Convierte la probabilidad alcista del modelo direccional a escala 0-100.
    model_probability debe venir como valor entre 0 y 1.
    """
    if model_probability is None or np.isnan(model_probability):
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

    Componentes:
    - Precio vs MA20
    - Precio vs MA50
    - MA20 vs MA50
    - RSI
    """

    score = 50.0

    # Tendencia de corto plazo
    if last_price > ma20:
        score += 12.5
    else:
        score -= 12.5

    # Tendencia de mediano plazo
    if last_price > ma50:
        score += 12.5
    else:
        score -= 12.5

    # Cruce de medias
    if ma20 > ma50:
        score += 12.5
    else:
        score -= 12.5

    # RSI
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

    La volatilidad entra como volatilidad relativa:
    std20 / precio_actual.

    Menor volatilidad = mayor score.
    Mayor volatilidad = menor score.
    """

    if volatility is None or np.isnan(volatility):
        return 50.0

    # Escala simple:
    # 0.00 = 100
    # 0.05 = 0
    volatility_score = 100 - (volatility / 0.05) * 100

    return float(np.clip(volatility_score, 0, 100))


def calculate_sentiment_score(
    sentiment_score: float | None = None
) -> float:
    """
    Placeholder para sentimiento financiero.

    Por ahora usamos 50 = neutral.
    Luego FinBERT alimentará este valor.
    """

    if sentiment_score is None or np.isnan(sentiment_score):
        return 50.0

    return float(np.clip(sentiment_score, 0, 100))


def classify_market_signal(score: float) -> tuple[str, str]:
    """
    Clasifica el score integrado en una señal ejecutiva.
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
    model_probability: float | None = None,
    sentiment_score: float | None = None
) -> dict:
    """
    Calcula el Market Signal Score integrado.

    Fórmula:
    40% probabilidad del modelo direccional
    25% score técnico
    20% sentimiento financiero
    15% ajuste por volatilidad
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
