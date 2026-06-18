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

## Resumen ejecutivo

Market Intelligence Dashboard es un proyecto de aprendizaje automático financiero aplicado, diseñado para evaluar si el sentimiento financiero extraído con FinBERT puede mejorar la predicción direccional a corto plazo de los activos financieros en comparación con modelos basados únicamente en indicadores técnicos.

El sistema integra datos históricos de mercado, indicadores técnicos, modelos de aprendizaje automático supervisado, backtesting, validación cruzada, pruebas de hipótesis estadísticas, análisis de sentimiento basado en FinBERT e IA explicable.

El proyecto no pretende ofrecer asesoramiento financiero. Su propósito es educativo, analítico y académico: demostrar un flujo de trabajo completo de principio a fin para la ciencia de datos financieros, la validación de modelos y la experimentación orientada a la investigación.

Pregunta de investigación

¿Mejora significativamente la incorporación del sentimiento financiero extraído mediante FinBERT la predicción direccional de la rentabilidad a corto plazo en comparación con modelos basados únicamente en indicadores técnicos?

Hipótesis principal

- H0: El sentimiento de FinBERT no mejora significativamente el rendimiento predictivo en comparación con el modelo basado únicamente en indicadores técnicos.

- H1: El sentimiento de FinBERT mejora significativamente el rendimiento predictivo en comparación con el modelo basado únicamente en indicadores técnicos.

---

## Estado actual de la investigación

Este proyecto implementa un pipeline aplicado de Financial Machine Learning para evaluar si el sentimiento financiero extraído con FinBERT mejora la predicción direccional de retornos de corto plazo frente a modelos basados únicamente en indicadores técnicos.

En la etapa actual, el sistema permite ejecutar inferencia real con FinBERT mediante la carga de un archivo CSV de noticias financieras. Sin embargo, las conclusiones académicas finales requieren un dataset histórico de noticias más amplio, documentado y temporalmente alineado con el activo financiero y el período de precios analizado.

El Market Signal Score incluido en el dashboard debe interpretarse como un indicador heurístico de producto. No se utiliza como evidencia científica para validar la hipótesis central, y sus pesos deben considerarse supuestos configurables pendientes de calibración empírica.

## English summary

This project implements an applied Financial Machine Learning pipeline to evaluate whether financial sentiment extracted with FinBERT improves short-term directional return forecasting compared with technical-indicator-only models.

At the current stage, the system supports real FinBERT inference through uploaded financial news CSV files. However, final academic conclusions require a larger, documented and temporally aligned historical news dataset.

The Market Signal Score included in the dashboard is a product-oriented heuristic indicator. It is not used as scientific evidence for the central research hypothesis, and its weights should be interpreted as configurable assumptions pending empirical calibration.

---

## Posicionamiento del proyecto

Este proyecto no busca reemplazar plataformas profesionales como Bloomberg Terminal, Refinitiv, FactSet o TradingView Premium.

Su valor principal está en la transparencia metodológica y en la reproducibilidad del experimento. A diferencia de plataformas comerciales orientadas al monitoreo de mercado, visualización o ejecución, este proyecto se enfoca en:

- Validación temporal mediante walk-forward.
- Comparación contra baselines académicos.
- Evaluación estadística de modelos predictivos.
- Integración experimental de sentimiento financiero con FinBERT.
- Documentación explícita de limitaciones.
- Interpretabilidad mediante Explainable AI.
- Implementación abierta en Python.

El objetivo no es vender señales de trading, sino demostrar cómo estructurar, validar y documentar un experimento de machine learning financiero de forma transparente.

English summary

This project is not intended to replace professional platforms such as Bloomberg Terminal, Refinitiv, FactSet or TradingView Premium.

Its main value lies in methodological transparency and experiment reproducibility. Unlike commercial platforms focused on market monitoring, charting or execution, this project emphasizes temporal validation, academic baselines, statistical testing, FinBERT-based sentiment experimentation, explicit limitations and Explainable AI.

The objective is not to sell trading signals, but to demonstrate how financial machine learning experiments can be structured, validated and documented transparently.

--+

## Estado de madurez

| Componente | Estado actual | Próxima mejora |
|---|---|---|
| Dashboard Streamlit | Funcional | Mejorar UI y experiencia de usuario |
| Datos de mercado | Funcional con yfinance | Documentar cobertura y limitaciones |
| Indicadores técnicos | Funcional | Agregar pruebas econométricas |
| Modelos ML | Funcional | Reportar métricas finales por activo |
| Backtesting | Funcional | Validar costos y supuestos |
| FinBERT | Funcional vía CSV cargado por usuario | Construir dataset histórico alineado |
| Pruebas estadísticas | Funcional | Ejecutar experimento completo con datos reales |
| Explainable AI | Funcional con feature importance | Agregar SHAP posteriormente |
| Research Summary | Funcional | Conectar con paper LaTeX |

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
| Página web del proyecto | [Project Website](https://sergiomendez2026.github.io/market-intelligence-dashboard/) | Presenta el proyecto de forma ejecutiva: objetivo, stack, funcionalidades, capturas y propuesta de valor. |
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
  https://sergiomendez2026.github.io/market-intelligence-dashboard/

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

## Arquitectura del proyecto

```text
market-intelligence-dashboard/
│
├── app.py                         # Aplicación principal en Streamlit
├── requirements.txt               # Dependencias de producción
├── requirements-dev.txt           # Dependencias de desarrollo y testing
├── README.md                      # Documentación principal del proyecto
├── index.html                     # Landing page bilingüe en GitHub Pages
│
├── src/
│   ├── data_loader.py             # Carga de datos financieros
│   ├── indicators.py              # Indicadores técnicos
│   ├── features.py                # Ingeniería de variables
│   ├── model.py                   # Modelos de regresión y clasificación
│   ├── signals.py                 # Market Signal Score
│   ├── backtesting.py             # Backtesting de estrategias
│   ├── baselines.py               # Baselines académicos
│   ├── walkforward_academic.py    # Validación walk-forward
│   ├── finbert_sentiment.py       # Pipeline de sentimiento financiero
│   ├── statistical_tests.py       # Pruebas estadísticas
│   ├── explainability.py          # Interpretabilidad de modelos
│   ├── portfolio.py               # Analítica de portafolio
│   └── ui.py                      # Componentes visuales
│
├── tests/                         # Pruebas unitarias
├── docs/                          # Documentación metodológica
├── reports/                       # Reportes y model cards
├── research/                      # Material académico del experimento
│
└── assets/
    ├── screenshots/               # Capturas del dashboard
    ├── diagrams/                  # Diagramas del pipeline
    └── social/                    # Imágenes para LinkedIn y GitHub
```

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

## Methodology

The dashboard follows a research-oriented financial machine learning workflow:

| Stage | Description |
|---|---|
| Market Data | Historical asset prices are loaded using `yfinance`. |
| Technical Indicators | Moving averages, RSI, volatility, returns and momentum variables are generated. |
| Machine Learning | Regression and directional classification models are trained. |
| Baselines | Naive, Linear Regression and ARIMA baselines are included for academic comparison. |
| Walk-Forward Validation | Models are evaluated through rolling temporal windows to reduce overfitting. |
| FinBERT Sentiment | Financial news are classified as positive, negative or neutral using FinBERT. |
| Statistical Tests | Diebold-Mariano, McNemar and bootstrap tests are applied to compare models. |
| Explainable AI | Feature importance is used to interpret model behavior. |
| Backtesting | Strategy performance is compared against Buy & Hold including transaction costs. |

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

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Market Data | yfinance |
| Machine Learning | Scikit-learn, XGBoost |
| Time Series | Statsmodels / ARIMA |
| NLP / Sentiment | Transformers, FinBERT, PyTorch |
| Statistical Testing | SciPy |
| Testing | Pytest |
| Deployment | Streamlit Cloud, GitHub |
| Documentation | Markdown, GitHub Pages |

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

## Current Results

The project currently demonstrates a complete working pipeline:

- Technical indicators are generated from historical price data.
- Machine learning models are compared against academic baselines.
- Walk-forward validation is implemented to simulate realistic temporal evaluation.
- FinBERT classifies financial news and converts sentiment into model features.
- Statistical tests compare the technical-only model against the sentiment-enhanced model.
- Explainable AI provides feature importance to support model interpretation.

Initial tests show that the technical + sentiment model can be evaluated against the technical-only model. However, final academic conclusions require a larger and temporally aligned financial news dataset.

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
