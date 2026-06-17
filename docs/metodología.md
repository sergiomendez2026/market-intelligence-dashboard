# Metodología

## 1. Alcance del proyecto

Market Intelligence Dashboard es una plataforma analítica académica/profesional para evaluar activos financieros mediante indicadores técnicos, modelos de machine learning, señales integradas, comparación de modelos, validación temporal y backtesting vectorizado con costos transaccionales.

El objetivo del proyecto no es prometer rentabilidad ni entregar asesoría financiera. El objetivo es construir un marco analítico reproducible que permita:

- Descargar datos históricos de activos financieros.
- Calcular indicadores técnicos.
- Construir variables predictivas para series temporales financieras.
- Entrenar modelos supervisados para predicción direccional.
- Comparar modelos contra baselines simples.
- Evaluar desempeño mediante métricas cuantitativas.
- Ejecutar validación walk-forward para reducir sobreajuste.
- Simular estrategias mediante backtesting.
- Presentar resultados en un dashboard ejecutivo e interactivo.

---

## 2. Fuente de datos

La versión actual utiliza datos públicos de mercado compatibles con "yfinance".

Activos soportados:

Activo| Ticker
Apple| "AAPL"
Tesla| "TSLA"
Bitcoin| "BTC-USD"
Ethereum| "ETH-USD"
S&P 500| "^GSPC"
NASDAQ| "^IXIC"
EUR/USD| "EURUSD=X"

Horizontes históricos disponibles:

Horizonte| Interpretación
"1y"| Último año histórico
"2y"| Últimos dos años
"5y"| Últimos cinco años

---

## 3. Indicadores técnicos

La plataforma calcula indicadores técnicos usados como variables explicativas para los modelos y para la construcción de señales de mercado.

Indicadores actuales:

- Media móvil de 20 períodos.
- Media móvil de 50 períodos.
- Bandas de Bollinger.
- RSI.
- Volatilidad móvil.
- Retornos históricos.
- Variables de momentum.
- Distancia relativa frente a medias móviles.

Estos indicadores permiten representar tendencia, momentum, volatilidad y posición relativa del precio.

---

## 4. Dataset de machine learning

El dataset de machine learning se construye a partir de precios históricos e indicadores derivados.

Objetivos actuales:

- Objetivo de regresión: precio futuro.
- Objetivo direccional: clasificación binaria de movimiento alcista/bajista.

Definición conceptual:

target_direction = 1 if future_price > current_price else 0

Esto permite evaluar si el modelo captura dirección de mercado, no solo error de predicción de precio.

---

## 5. Modelos evaluados

La plataforma compara modelos supervisados para predicción direccional.

Modelos actuales:

- Baseline Dummy.
- Logistic Regression.
- Random Forest.
- XGBoost.

El baseline se usa como referencia mínima. En mercados financieros, un modelo solo tiene valor metodológico si supera de manera consistente una regla simple de comparación.

---

## 6. Métricas de evaluación

Métricas reportadas:

Métrica| Descripción
Accuracy| Proporción de predicciones direccionales correctas.
Precision| Proporción de señales alcistas predichas que fueron correctas.
Recall| Proporción de movimientos alcistas reales detectados.
F1 Score| Media armónica entre precision y recall.
Robustness Score| Métrica compuesta para priorizar modelos con desempeño equilibrado.
Exceso vs baseline| Diferencia de desempeño frente al modelo base.

El F1 Score es relevante cuando existe desbalance de clases. El Robustness Score ayuda a evitar seleccionar modelos únicamente por accuracy.

---

## 7. Validación walk-forward

La validación walk-forward entrena el modelo usando únicamente datos pasados y evalúa en ventanas futuras.

Este enfoque reduce look-ahead bias y simula mejor un escenario real de uso en series temporales financieras.

Métricas reportadas:

- MAE walk-forward.
- MAE baseline.
- MAPE.
- Dirección correcta.
- Accuracy direccional.
- Precision.
- Recall.
- F1 Score.
- Baseline direccional.
- Observaciones evaluadas.

La validación walk-forward es computacionalmente más costosa que una división simple 80/20, pero metodológicamente es más rigurosa para series temporales.

---

## 8. Market Signal Score

El Market Signal Score integra cuatro componentes:

market_signal_score =
    0.40 * model_probability_score
  + 0.25 * technical_score
  + 0.20 * sentiment_score
  + 0.15 * volatility_score

Componentes del score

Factor| Peso| Descripción
"model_probability_score"| 40%| Probabilidad alcista estimada por el modelo direccional.
"technical_score"| 25%| Evaluación basada en medias móviles, RSI, tendencia y momentum.
"sentiment_score"| 20%| Señal de sentimiento financiero. Actualmente usa fallback liviano; FinBERT puede integrarse como capa avanzada.
"volatility_score"| 15%| Ajuste por riesgo basado en volatilidad relativa.

Clasificación de señal

Rango del score| Señal| Interpretación
80 - 100| Strong Bullish| Señal alcista fuerte.
60 - 79.99| Bullish moderado| Señal positiva moderada.
40 - 59.99| Neutral| Señales mixtas o sin dirección dominante.
20 - 39.99| Bearish moderado| Señal negativa moderada.
0 - 19.99| Strong Bearish| Señal bajista fuerte.

Consideraciones metodológicas

El score debe interpretarse como una métrica analítica, no como una recomendación financiera. Sus pesos son definidos de forma heurística y pueden calibrarse en futuras versiones mediante validación histórica, optimización bayesiana o búsqueda de hiperparámetros.

El componente de sentimiento se mantiene neutral cuando no existe suficiente información textual confiable. Esto evita introducir ruido artificial en la señal.

---

## 9. Backtesting

La plataforma incluye backtesting vectorizado con supuestos explícitos.

El módulo actual reporta:

- Capital inicial simulado.
- Benchmark Buy & Hold.
- Curva de capital de la estrategia.
- Comisiones por operación.
- Slippage estimado.
- Número de operaciones.
- Costo transaccional total.
- Exposición al mercado.
- Sharpe Ratio aproximado.
- Retorno de estrategia.
- Retorno Buy & Hold.
- Exceso de retorno.
- Drawdown de estrategia.

La señal se desplaza un período para reducir look-ahead bias.

---

## 10. Limitaciones metodológicas

Limitaciones actuales:

- Los datos provienen de fuentes públicas.
- No incluye datos fundamentales corporativos.
- No incluye datos intradía.
- No incluye datos de libro de órdenes.
- No modela spread dinámico.
- No considera profundidad real de mercado.
- No considera impuestos.
- No considera restricciones reales de liquidez.
- El sentimiento financiero usa fallback liviano.
- El backtesting es vectorizado y no representa ejecución real de mercado.
- Los resultados pueden variar según activo, período y régimen de mercado.
- Los modelos pueden no superar al baseline en ciertos escenarios.

---

## 11. Reproducibilidad

El proyecto incluye:

- Código modular en "src/".
- Tests unitarios en "tests/".
- GitHub Actions.
- "requirements.txt" para ejecución.
- "requirements-dev.txt" para desarrollo y testing.
- Licencia MIT.
- Changelog.
- README técnico.
- Documentación extendida en "docs/".

---

## 12. Disclaimer

Este proyecto tiene fines educativos, analíticos y demostrativos. No constituye asesoría financiera, recomendación de inversión ni promesa de rentabilidad.
