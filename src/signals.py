# src/signals.py

import numpy as np


def compute_technical_score(rsi: float, price: float, ma20: float, ma50: float) -> float:
    """
    Calcula un score técnico entre 0 y 1.
    Valores altos indican sesgo alcista.
    """

    score = 0.5

    # RSI
    if rsi < 30:
        score += 0.20
    elif rsi > 70:
        score -= 0.20

    # Tendencia por medias móviles
    if price > ma20:
        score += 0.15
    else:
        score -= 0.15

    if ma20 > ma50:
        score += 0.15
    else:
        score -= 0.15

    return float(np.clip(score, 0, 1))


def compute_model_score(last_price: float, predicted_price: float) -> float:
    """
    Convierte la predicción del modelo en score entre 0 y 1.
    """

    expected_return = (predicted_price - last_price) / last_price

    if expected_return > 0.03:
        return 0.85
    elif expected_return > 0.01:
        return 0.70
    elif expected_return > -0.01:
        return 0.50
    elif expected_return > -0.03:
        return 0.30
    else:
        return 0.15


def compute_volatility_score(volatility: float) -> float:
    """
    Penaliza activos con volatilidad elevada.
    """

    if volatility <= 0:
        return 0.5

    penalty = min(volatility * 10, 0.5)

    return float(np.clip(1 - penalty, 0, 1))


def compute_market_signal(
    last_price: float,
    predicted_price: float,
    rsi: float,
    ma20: float,
    ma50: float,
    volatility: float
) -> dict:
    """
    Combina modelo, indicadores técnicos y volatilidad para generar una señal ejecutiva.
    """

    model_score = compute_model_score(last_price, predicted_price)
    technical_score = compute_technical_score(rsi, last_price, ma20, ma50)
    volatility_score = compute_volatility_score(volatility)

    final_score = (
        0.45 * model_score +
        0.40 * technical_score +
        0.15 * volatility_score
    )

    signal_score = round(final_score * 100, 2)

    if signal_score >= 70:
        signal = "Bullish fuerte"
        interpretation = "El activo muestra una señal alcista fuerte según el modelo, tendencia técnica y volatilidad."
    elif signal_score >= 55:
        signal = "Bullish moderado"
        interpretation = "El activo muestra un sesgo alcista moderado, aunque requiere monitoreo."
    elif signal_score >= 45:
        signal = "Neutral"
        interpretation = "El activo no muestra una señal dominante. El escenario es mixto."
    elif signal_score >= 30:
        signal = "Bearish moderado"
        interpretation = "El activo muestra debilidad moderada según las señales actuales."
    else:
        signal = "Bearish fuerte"
        interpretation = "El activo muestra una señal bajista fuerte según el modelo y los indicadores."

    return {
        "signal_score": signal_score,
        "signal": signal,
        "interpretation": interpretation,
        "model_score": round(model_score * 100, 2),
        "technical_score": round(technical_score * 100, 2),
        "volatility_score": round(volatility_score * 100, 2),
    }
