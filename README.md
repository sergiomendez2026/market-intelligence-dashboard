# Market Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Release](https://img.shields.io/github/v/release/sergiomendez2026/market-intelligence-dashboard)
![Tests](https://github.com/sergiomendez2026/market-intelligence-dashboard/actions/workflows/python-tests.yml/badge.svg)
![Status](https://img.shields.io/badge/status-academic%2Fprofessional%20MVP-success)

Plataforma analítica para evaluación de activos financieros mediante indicadores técnicos, modelos de machine learning, señales integradas, comparación de modelos, validación temporal y backtesting con costos transaccionales.

Proyecto académico/profesional con fines educativos y analíticos.
No constituye asesoría financiera, recomendación de inversión ni promesa de rentabilidad.

---

## Executive Summary (English)

Market Intelligence Dashboard is an applied financial machine learning project designed to evaluate whether financial sentiment extracted with FinBERT can improve short-term directional forecasting of financial assets compared with models based only on technical indicators.

The system integrates historical market data, technical indicators, supervised machine learning models, backtesting, walk-forward validation, statistical hypothesis testing, FinBERT-based sentiment analysis and Explainable AI.

The project is not intended to provide financial advice. Its purpose is educational, analytical and academic: to demonstrate a complete end-to-end workflow for financial data science, model validation and research-oriented experimentation.

Research Question

Does the incorporation of financial sentiment extracted through FinBERT significantly improve short-term directional return forecasting compared with models based only on technical indicators?

Core Hypothesis

- H0: FinBERT sentiment does not significantly improve predictive performance compared with the technical-only model.
- H1: FinBERT sentiment significantly improves predictive performance compared with the technical-only model.

---

Resumen ejecutivo

Market Intelligence Dashboard es un proyecto de aprendizaje automático financiero aplicado, diseñado para evaluar si el sentimiento financiero extraído con FinBERT puede mejorar la predicción direccional a corto plazo de los activos financieros en comparación con modelos basados únicamente en indicadores técnicos.

El sistema integra datos históricos de mercado, indicadores técnicos, modelos de aprendizaje automático supervisado, backtesting, validación cruzada, pruebas de hipótesis estadísticas, análisis de sentimiento basado en FinBERT e IA explicable.

El proyecto no pretende ofrecer asesoramiento financiero. Su propósito es educativo, analítico y académico: demostrar un flujo de trabajo completo de principio a fin para la ciencia de datos financieros, la validación de modelos y la experimentación orientada a la investigación.

Pregunta de investigación

¿Mejora significativamente la incorporación del sentimiento financiero extraído mediante FinBERT la predicción direccional de la rentabilidad a corto plazo en comparación con modelos basados únicamente en indicadores técnicos?

Hipótesis principal

- H0: El sentimiento de FinBERT no mejora significativamente el rendimiento predictivo en comparación con el modelo basado únicamente en indicadores técnicos.

- H1: El sentimiento de FinBERT mejora significativamente el rendimiento predictivo en comparación con el modelo basado únicamente en indicadores técnicos.

---

## 1. Objetivo del proyecto

El objetivo de este proyecto es construir un sistema modular de inteligencia de mercado capaz de:

- Descargar datos históricos de activos financieros.
- Calcular indicadores técnicos relevantes.
- Construir variables predictivas para series temporales financieras.
- Entrenar modelos supervisados para predicción direccional.
- Comparar modelos contra baselines simples.
- Generar un Market Signal Score integrado.
- Ejecutar backtesting con comisiones y slippage.
- Aplicar validación walk-forward para reducir sobreajuste.
- Presentar resultados en un dashboard interactivo desplegado en la nube.

El proyecto prioriza rigor metodológico sobre promesas de rentabilidad. En mercados financieros, superar consistentemente un baseline simple es difícil; por eso el sistema reporta explícitamente cuando un modelo no supera al benchmark.

---

## Ecosistema del proyecto

Este proyecto está estructurado como un ecosistema profesional de tres capas:

| Capa | Enlace | Propósito |
|---|---|---|
| Repositorio GitHub | [Código fuente](https://github.com/sergiomendez2026/market-intelligence-dashboard) | Contiene la arquitectura modular, código Python, modelos, validación, backtesting y documentación técnica. |
| Página web del proyecto | [Project Website](https://sergiomendez2026.github.io/market-intelligence-dashboard/#stack) | Presenta el proyecto de forma ejecutiva: objetivo, stack, funcionalidades, capturas y propuesta de valor. |
| Dashboard interactivo | [Live Dashboard](https://market-intelligence-dashboard-fjy5vx69qtam5vpxxqcdno.streamlit.app/) | Permite probar la solución en tiempo real con activos financieros, indicadores, modelos, señales y backtesting. |

La lógica del ecosistema es:

---

GitHub Repository
↓
Evidencia técnica: código, arquitectura, metodología y documentación.

Project Website
↓
Presentación ejecutiva: problema, solución, stack, capturas y valor profesional.

Interactive Dashboard
↓
Demo funcional: análisis financiero, modelos, señales integradas y backtesting.

---

## 2. Enlaces del proyecto

- Repositorio GitHub:
  https://github.com/sergiomendez2026/market-intelligence-dashboard

- Página web del proyecto:
  https://sergiomendez2026.github.io/market-intelligence-dashboard/#stack

- Dashboard interactivo:
  https://market-intelligence-dashboard-fjy5vx69qtam5vpxxqcdno.streamlit.app/

---

## 3. Activos analizados

Actualmente la aplicación permite analizar:

- Apple — AAPL
- Tesla — TSLA
- Bitcoin — BTC-USD
- Ethereum — ETH-USD
- S&P 500 — ^GSPC
- NASDAQ — ^IXIC
- EUR/USD — EURUSD=X

Los datos son obtenidos desde fuentes financieras públicas compatibles con "yfinance".

---

## 4. Arquitectura del sistema

```text
market-intelligence-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── features.py
│   ├── model.py
│   ├── model_comparison.py
│   ├── signals.py
│   ├── sentiment.py
│   ├── financial_metrics.py
│   ├── backtesting.py
│   ├── validation.py
│   └── ui.py
│
├── assets/
├── notebooks/
├── reports/
└── tests/
```

La arquitectura separa responsabilidades para mejorar mantenibilidad, escalabilidad y claridad técnica.

| Módulo | Responsabilidad |
|---|---|
| `app.py` | Orquestación principal de la aplicación Streamlit. |
| `data_loader.py` | Carga de datos históricos desde fuentes financieras. |
| `indicators.py` | Cálculo de indicadores técnicos como medias móviles, RSI y Bollinger Bands. |
| `features.py` | Construcción del dataset de machine learning. |
| `model.py` | Entrenamiento y evaluación de modelos predictivos. |
| `model_comparison.py` | Comparación de modelos direccionales contra baseline. |
| `signals.py` | Cálculo del Market Signal Score integrado. |
| `sentiment.py` | Análisis de sentimiento financiero con fallback liviano. |
| `financial_metrics.py` | Cálculo de KPIs financieros: retorno, volatilidad y drawdown. |
| `backtesting.py` | Simulación de estrategia con comisiones y slippage. |
| `validation.py` | Validación walk-forward para series temporales. |
| `ui.py` | Componentes visuales, sidebar, disclaimers y UX ejecutiva. |

---

## 5. Indicadores financieros y técnicos

El sistema calcula:

- Precio actual
- Máximo y mínimo del período
- Retorno acumulado
- Volatilidad anualizada
- Drawdown máximo
- Media móvil de 20 períodos
- Media móvil de 50 períodos
- Bandas de Bollinger
- RSI de 14 períodos
- Retornos rezagados
- Volatilidades móviles
- Momentum
- Distancia relativa frente a medias móviles

---

## 6. Machine Learning

El sistema usa modelos supervisados para evaluar señales direccionales sobre activos financieros.

Modelos implementados

- Baseline Dummy
- Logistic Regression
- Random Forest
- XGBoost

Target direccional

El objetivo direccional se define como:

target_direction = 1 si el precio futuro supera al precio actual
target_direction = 0 en caso contrario

El horizonte puede ajustarse para evaluar movimientos del siguiente período o ventanas futuras.

---

## 7. Comparación de modelos

La aplicación compara modelos usando:

- Accuracy
- Precision
- Recall
- F1 Score
- Exceso de accuracy contra baseline
- Robustness Score

El sistema evita seleccionar modelos únicamente por accuracy, ya que en clasificación financiera puede existir desbalance de clases. Por eso también se reportan F1 Score y Robustness Score.

---

## 8. Market Signal Score

El `Market Signal Score` es una señal ejecutiva integrada que resume cuatro fuentes de información:

1. Probabilidad alcista estimada por el modelo direccional.
2. Condición técnica del activo.
3. Sentimiento financiero.
4. Penalización por volatilidad.

El objetivo no es predecir el mercado con certeza, sino construir una métrica compuesta, interpretable y auditable para comparar activos bajo criterios homogéneos.

```text
market_signal_score =
    0.40 * model_probability_score
  + 0.25 * technical_score
  + 0.20 * sentiment_score
  + 0.15 * volatility_score
```

### Componentes del score

| Factor | Peso | Descripción |
|---|---:|---|
| `model_probability_score` | 40% | Probabilidad alcista estimada por el modelo direccional. |
| `technical_score` | 25% | Evaluación basada en medias móviles, RSI, tendencia y momentum. |
| `sentiment_score` | 20% | Señal de sentimiento financiero. Actualmente usa fallback liviano; FinBERT puede integrarse como capa avanzada. |
| `volatility_score` | 15% | Ajuste por riesgo. Penaliza activos con mayor volatilidad relativa. |

### Interpretación

| Rango del score | Señal | Interpretación |
|---:|---|---|
| 80 - 100 | Strong Bullish | Señal alcista fuerte. |
| 60 - 79.99 | Bullish moderado | Señal positiva moderada. |
| 40 - 59.99 | Neutral | Señales mixtas o sin dirección dominante. |
| 20 - 39.99 | Bearish moderado | Señal negativa moderada. |
| 0 - 19.99 | Strong Bearish | Señal bajista fuerte. |

### Consideraciones metodológicas

El score debe interpretarse como una métrica analítica, no como una recomendación financiera. Sus pesos son definidos de forma heurística y pueden calibrarse en futuras versiones mediante validación histórica, optimización bayesiana o búsqueda de hiperparámetros.

El componente de sentimiento se mantiene neutral cuando no existe suficiente información textual confiable. Esto evita introducir ruido artificial en la señal final.

---

## 9. Backtesting

El sistema incluye backtesting vectorizado con:

- Capital inicial simulado
- Buy & Hold como benchmark
- Estrategia basada en señales
- Comisiones por operación
- Slippage estimado
- Número de operaciones
- Costo transaccional total
- Retorno de estrategia
- Retorno Buy & Hold
- Exceso de retorno
- Drawdown de estrategia
- Exposición al mercado
- Sharpe Ratio aproximado

La señal se desplaza un período para reducir look-ahead bias.

---

## 10. Validación walk-forward

La validación walk-forward entrena modelos usando únicamente datos pasados y evalúa en ventanas futuras.

Esto permite simular un escenario más cercano al uso real del modelo en series temporales financieras.

La aplicación reporta:

- MAE walk-forward
- MAE baseline
- MAPE
- Dirección correcta
- Accuracy direccional
- Precision
- Recall
- F1 Score
- Baseline direccional
- Observaciones evaluadas

---

## 11. Sentimiento financiero

El sistema incluye una capa de sentimiento financiero con fallback liviano basado en palabras clave.

La arquitectura está preparada para integrar FinBERT como capa avanzada de NLP financiero.

Estado actual:

- Fallback liviano activo.
- FinBERT planificado como extensión avanzada.
- Score neutral usado cuando no existe señal textual suficiente.

---

## 12. Stack tecnológico

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- XGBoost
- yfinance
- GitHub
- GitHub Pages
- Streamlit Cloud

---

## 13. Limitaciones metodológicas

Este proyecto no promete rentabilidad. Sus principales limitaciones actuales son:

- Los datos provienen de fuentes públicas.
- No incluye datos fundamentales corporativos.
- El sentimiento financiero aún usa fallback liviano.
- El backtesting es vectorizado y no representa ejecución real de mercado.
- Los resultados pueden cambiar según activo, período y régimen de mercado.
- Los modelos pueden no superar al baseline en ciertos escenarios.
- No se consideran restricciones reales de liquidez, impuestos ni spread dinámico.

---

## 14. Potencial de mejora

Próximas extensiones:

- Integración completa con FinBERT.
- Optimización de hiperparámetros.
- Modelos LightGBM, CatBoost, LSTM, Temporal Fusion Transformer.
- Backtesting por portafolio.
- Optimización de carteras.
- Alertas automáticas por email, Telegram o WhatsApp.
- Generación automática de reportes PDF.
- API con FastAPI.
- Frontend profesional con React o Next.js.
- Sistema multiusuario con autenticación.
- Versión multilingüe español/inglés.

---

## 15. Valor académico y profesional

Este proyecto demuestra competencias en:

- Ciencia de datos aplicada.
- Machine learning sobre series temporales.
- Finanzas cuantitativas.
- Ingeniería de software modular.
- Visualización de datos.
- Validación rigurosa de modelos.
- Diseño de productos analíticos.
- Despliegue cloud.
- Comunicación técnica mediante documentación y dashboard.

---

## 16. Testing

El proyecto incluye pruebas unitarias básicas para validar componentes críticos del sistema:

- Cálculo del Market Signal Score.
- Métricas financieras.
- Rango válido de señales.
- Estructura esperada de salidas.

---

## 17. CI/CD y testing automático

El proyecto incluye un workflow de GitHub Actions para ejecutar pruebas automáticas sobre el código Python cada vez que se realiza un cambio en el repositorio.

Archivo de workflow:

.github/workflows/python-tests.yml

## 18. Ejecución local

git clone 

https://github.com/sergiomendez2026/market-intelligence-dashboard.git

cd market-intelligence-dashboard

pip install -r requirements.txt

streamlit run app.py

---

## 19. Versioning

This repository follows a simple semantic versioning structure:

| Version | Status | Description |
|---|---|---|
| v0.1.0 | Release Candidate | First functional academic/professional version |

Current version:

v0.1.0

---

## 20. Disclaimer

---

Este proyecto tiene fines educativos, analíticos y demostrativos.

No constituye asesoría financiera, recomendación de inversión, señal de trading, asesoría legal ni promesa de rentabilidad.

Los resultados generados por la aplicación deben interpretarse como salidas experimentales de un sistema de análisis cuantitativo y no como instrucciones para comprar, vender o mantener activos financieros.

## 21. Roadmap

Planned improvements:

- Full FinBERT integration for financial sentiment analysis.
- Portfolio-level backtesting.
- Hyperparameter optimization.
- Model calibration and probability reliability analysis.
- Risk-adjusted portfolio optimization.
- Automatic PDF report generation.
- API backend with FastAPI.
- Professional frontend with React or Next.js.
- Multi-user authentication.
- Multilingual version: Spanish and English.

## Enfoque académico de investigación

Además de funcionar como dashboard profesional, este proyecto se desarrolla como una investigación aplicada en machine learning financiero.

La pregunta central es:

> ¿La incorporación de sentimiento financiero extraído mediante FinBERT mejora significativamente la predicción direccional de retornos de corto plazo frente a modelos basados únicamente en indicadores técnicos?

El diseño académico completo se encuentra en:

- [Diseño de investigación](docs/diseno_investigacion.md)

La versión LaTeX del documento académico se encuentra en:

- [Plantilla LaTeX](research/latex/main.tex)

## Documentación técnica extendida

La documentación técnica ampliada del proyecto se encuentra en la carpeta `docs/`:

- [Metodología](docs/metodologia.md): metodología analítica, construcción del score, validación, backtesting y limitaciones.
- [Arquitectura](docs/arquitectura.md): estructura modular, flujo de datos y responsabilidades por módulo.
- [Roadmap](docs/roadmap.md): fases de evolución técnica, académica y potencial comercial.

## Development setup

Install production dependencies:

```bash
pip install -r requirements.txt
