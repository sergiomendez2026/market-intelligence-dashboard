# Diseño de investigación

## 1. Título tentativo

Evaluación del impacto del sentimiento financiero basado en FinBERT sobre la predicción direccional de retornos de corto plazo en activos financieros mediante modelos de machine learning y validación walk-forward.

---

## 2. Planteamiento del problema

La predicción de precios y retornos financieros es un problema complejo debido a la naturaleza ruidosa, no estacionaria y altamente competitiva de los mercados. De acuerdo con la hipótesis de eficiencia de mercado, la información disponible tiende a incorporarse rápidamente en los precios, lo que dificulta que modelos predictivos simples o complejos superen de forma consistente a estrategias ingenuas o baselines tradicionales.

En los últimos años, los modelos de machine learning han sido utilizados para identificar patrones no lineales en series temporales financieras. Sin embargo, muchos estudios y proyectos aplicados presentan limitaciones metodológicas: uso de un único split de entrenamiento/prueba, ausencia de baselines rigurosos, riesgo de look-ahead bias, falta de pruebas estadísticas y poca comparación entre modelos con y sin variables alternativas.

Una fuente adicional de información potencialmente relevante es el sentimiento financiero extraído de noticias, titulares o textos del mercado. Modelos NLP especializados como FinBERT permiten clasificar sentimiento financiero positivo, negativo o neutral. No obstante, la pregunta central no es si FinBERT puede clasificar textos, sino si su incorporación mejora significativamente la predicción de retornos o dirección de mercado frente a modelos basados únicamente en indicadores técnicos.

Este proyecto busca evaluar de manera rigurosa si el sentimiento financiero aporta valor predictivo incremental cuando se combina con indicadores técnicos y modelos supervisados, utilizando validación walk-forward, modelos baseline y pruebas estadísticas.

---

## 3. Pregunta de investigación

¿La incorporación de sentimiento financiero extraído mediante FinBERT mejora significativamente la predicción direccional de retornos de corto plazo frente a modelos basados únicamente en indicadores técnicos?

---

## 4. Hipótesis

Hipótesis nula

H0: La incorporación de sentimiento financiero mediante FinBERT no mejora significativamente el desempeño predictivo frente a modelos basados únicamente en indicadores técnicos.

Hipótesis alternativa

H1: La incorporación de sentimiento financiero mediante FinBERT mejora significativamente el desempeño predictivo frente a modelos basados únicamente en indicadores técnicos.

---

## 5. Objetivo general

Evaluar si la incorporación de sentimiento financiero basado en FinBERT mejora significativamente la predicción direccional de retornos de corto plazo en activos financieros, comparando modelos técnicos y modelos técnicos enriquecidos con sentimiento bajo un esquema de validación walk-forward.

---

## 6. Objetivos específicos

1. Construir un dataset financiero reproducible con precios históricos, retornos e indicadores técnicos.
2. Incorporar una capa de sentimiento financiero basada en FinBERT a partir de noticias o titulares financieros.
3. Implementar modelos baseline obligatorios para comparación.
4. Entrenar modelos técnicos sin sentimiento y modelos técnicos con sentimiento.
5. Evaluar el desempeño predictivo mediante métricas de clasificación y forecasting.
6. Aplicar validación walk-forward para reducir riesgo de look-ahead bias.
7. Comparar estadísticamente el desempeño entre modelos con y sin sentimiento.
8. Analizar las limitaciones metodológicas del enfoque.
9. Presentar resultados mediante un dashboard profesional y documentación académica.

---

## 7. Variables del estudio

Variable dependiente principal

La variable dependiente principal será la dirección futura del retorno:

target_direction = 1 si return(t+1) > 0
target_direction = 0 si return(t+1) <= 0

También se podrá evaluar como variable secundaria el retorno futuro:

future_return = return(t+1)

Variables independientes técnicas

Las variables técnicas incluirán:

- Retornos rezagados.
- Media móvil de 20 períodos.
- Media móvil de 50 períodos.
- Distancia relativa del precio frente a medias móviles.
- RSI.
- Bandas de Bollinger.
- Volatilidad móvil.
- Momentum.
- Drawdown.
- Variables de tendencia.

Variables independientes de sentimiento

Las variables de sentimiento incluirán:

- Sentimiento positivo.
- Sentimiento negativo.
- Sentimiento neutral.
- Score agregado de sentimiento.
- Sentimiento promedio por ventana temporal.
- Conteo de noticias por período.
- Intensidad del sentimiento.

---

## 8. Modelos obligatorios a comparar

El proyecto debe comparar cinco grupos de modelos.

Modelo 1: Naive

Predicción ingenua:

precio(t+1) = precio(t)

Para dirección:

dirección(t+1) = dirección(t)

Este modelo representa el baseline mínimo que cualquier modelo complejo debe superar.

Modelo 2: Regresión lineal

Modelo clásico con las mismas variables técnicas.

Objetivo:

- Evaluar si una relación lineal simple explica parte del comportamiento futuro.
- Servir como baseline interpretable frente a modelos más complejos.

Modelo 3: ARIMA/SARIMA

Modelo econométrico clásico para series temporales.

Objetivo:

- Comparar machine learning contra una familia tradicional de forecasting.
- Evaluar si la estructura temporal lineal aporta capacidad predictiva.

Modelo 4: Modelo técnico sin sentimiento

Modelo supervisado usando únicamente variables técnicas.

Ejemplos:

- Random Forest.
- XGBoost.
- Logistic Regression.
- Otros modelos supervisados.

Este modelo representa el sistema base.

Modelo 5: Modelo técnico + FinBERT

Modelo supervisado usando:

- Variables técnicas.
- Variables de sentimiento financiero generadas con FinBERT.

Este es el modelo experimental principal.

La comparación central será:

Modelo técnico sin sentimiento
vs
Modelo técnico + FinBERT

## Modelos baseline implementados

El proyecto incorpora los siguientes baselines académicos:

| Modelo | Propósito |
|---|---|
| Naive t+1 = precio actual | Baseline mínimo de forecasting financiero. |
| Regresión lineal | Baseline interpretable con variables técnicas. |
| ARIMA/SARIMA | Baseline econométrico clásico de series temporales. |
| Modelo técnico sin sentimiento | Modelo supervisado con indicadores técnicos. |
| Modelo técnico + FinBERT | Modelo experimental con información técnica y sentimiento financiero. |

La comparación principal de la investigación será entre el modelo técnico sin sentimiento y el modelo técnico + FinBERT.

---

## 9. Métrica principal

La métrica principal será:

F1 Score direccional bajo validación walk-forward

Se elige F1 Score porque la predicción direccional puede presentar desbalance entre clases alcistas y bajistas. Accuracy por sí sola puede ser engañosa si una clase domina el período analizado.

---

## 10. Métricas secundarias

Métricas de clasificación:

- Accuracy.
- Precision.
- Recall.
- F1 Score.
- Matriz de confusión.
- Balanced Accuracy.

Métricas de forecasting:

- MAE.
- RMSE.
- MAPE.
- Error frente al modelo naive.

Métricas financieras complementarias:

- Retorno acumulado de estrategia simulada.
- Sharpe Ratio.
- Maximum Drawdown.
- Exposición al mercado.
- Número de operaciones.
- Costos transaccionales.

---

## 11. Estrategia de validación

La validación principal será walk-forward validation.

El esquema general será:

1. Entrenar el modelo con una ventana histórica inicial.
2. Predecir el siguiente bloque temporal.
3. Avanzar la ventana.
4. Reentrenar el modelo.
5. Repetir el proceso hasta cubrir el período de evaluación.
6. Agregar los resultados de todas las ventanas.

Este enfoque evita entrenar con información futura y reduce el riesgo de look-ahead bias.

---

## 12. Pruebas estadísticas

Para comparar modelos se considerarán las siguientes pruebas:

Comparación de errores de forecasting

Se podrá usar el test Diebold-Mariano para comparar si las diferencias de error entre modelos son estadísticamente significativas.

Comparación principal:

Naive vs modelo técnico
Modelo técnico vs modelo técnico + FinBERT

Comparación de clasificación direccional

Para clasificación se podrá usar:

- McNemar test.
- Bootstrap sobre diferencias de F1 Score.
- Bootstrap sobre diferencias de accuracy.
- Intervalos de confianza para métricas direccionales.

El objetivo será determinar si la mejora observada es estadísticamente significativa y no solo producto del azar.

---

## 13. Alcance del estudio

El alcance inicial será limitado para mantener profundidad metodológica.

Activos sugeridos

Opción principal:

- Apple ("AAPL")

Opción alternativa:

- S&P 500 ("^GSPC")

Opción ampliada:

- Apple.
- Tesla.
- S&P 500.
- Bitcoin.

Horizonte de predicción

Horizonte principal:

t+1

Es decir, predicción del siguiente período.

Horizonte alternativo:

t+5

Predicción acumulada a cinco períodos.

Frecuencia

La frecuencia inicial será diaria.

---

## 14. Contribución esperada

La contribución esperada del proyecto es evaluar empíricamente si el sentimiento financiero especializado aporta valor incremental a modelos de predicción direccional de corto plazo.

El proyecto no busca demostrar que los mercados son fácilmente predecibles. Al contrario, busca producir evidencia rigurosa sobre si una fuente alternativa de información mejora o no mejora el desempeño frente a baselines razonables.

Un resultado negativo también sería válido. Si el modelo técnico + FinBERT no supera de forma significativa al modelo técnico sin sentimiento o al baseline naive, la conclusión sería metodológicamente relevante.

---

## 15. Limitaciones esperadas

Limitaciones principales:

- Los datos de precios provenientes de fuentes públicas pueden tener ajustes retroactivos.
- "yfinance" no garantiza datos point-in-time.
- Las noticias históricas pueden ser difíciles de obtener de forma gratuita.
- Existe riesgo de survivorship bias si se evalúan solo activos exitosos.
- Existe riesgo de look-ahead bias si el sentimiento no se alinea correctamente con la fecha real de publicación.
- El sentimiento textual puede no reflejar información nueva.
- El mercado puede haber incorporado la noticia antes de la fecha registrada.
- Los resultados pueden variar por activo, período y régimen de mercado.
- Una mejora estadística no implica rentabilidad real.
- Un backtest vectorizado no representa ejecución real de mercado.

---

## 16. Estructura académica propuesta

La investigación podrá organizarse así:

1. Resumen.
2. Introducción.
3. Revisión de literatura.
4. Metodología.
5. Datos.
6. Modelos.
7. Diseño experimental.
8. Resultados.
9. Discusión.
10. Limitaciones.
11. Conclusiones.
12. Referencias.

---

## 17. Relación con el dashboard profesional

El dashboard de Streamlit funcionará como capa de presentación del sistema experimental.

Debe mostrar:

- Activo analizado.
- Métricas principales.
- Comparación entre modelos.
- Resultado de validación walk-forward.
- Comparación técnico vs técnico + sentimiento.
- Backtesting complementario.
- Gráficos explicativos.
- Disclaimer financiero.

El dashboard no debe presentarse como herramienta de recomendación de inversión, sino como plataforma experimental de análisis cuantitativo.

---

## 18. Criterio de éxito académico

El proyecto alcanzará nivel académico alto si cumple con:

- Pregunta de investigación clara.
- Hipótesis explícitas.
- Revisión de literatura.
- Baselines obligatorios.
- Validación walk-forward.
- Pruebas estadísticas.
- Comparación con y sin FinBERT.
- Discusión honesta de resultados.
- Limitaciones explícitas.
- Reproducibilidad del código.

---

## 19. Criterio de éxito profesional

El proyecto alcanzará nivel profesional alto si cumple con:

- Código modular.
- Tests automatizados.
- GitHub Actions.
- README profesional.
- Documentación extendida.
- Dashboard estable.
- Visualización ejecutiva.
- Release versionado.
- Licencia.
- Roadmap.
- Explicabilidad del modelo.
- Métricas financieras claras.
- Disclaimer visible.

---

## 20. Disclaimer

Este proyecto tiene fines educativos, analíticos y demostrativos.

No constituye asesoría financiera, recomendación de inversión, señal de trading ni promesa de rentabilidad.
