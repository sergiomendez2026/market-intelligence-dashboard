# Changelog

Todos los cambios relevantes del proyecto serán documentados en este archivo.

El formato sigue una estructura simple basada en versiones semánticas.

---

## [v0.1.0] - 2026-06-17

### Agregado

- Dashboard interactivo desarrollado con Streamlit.
- Carga de datos financieros históricos mediante `yfinance`.
- Soporte inicial para activos financieros:
  - Apple
  - Tesla
  - Bitcoin
  - Ethereum
  - S&P 500
  - NASDAQ
  - EUR/USD
- Cálculo de indicadores técnicos:
  - Media móvil de 20 períodos.
  - Media móvil de 50 períodos.
  - Bandas de Bollinger.
  - RSI.
- KPIs financieros:
  - Precio actual.
  - Máximo histórico del período.
  - Mínimo histórico del período.
  - Retorno del período.
  - Volatilidad anualizada.
  - Drawdown máximo.
- Dataset de machine learning para series temporales financieras.
- Modelo de regresión para predicción de precio.
- Modelo direccional para clasificación alcista/bajista.
- Comparación de modelos:
  - Baseline Dummy.
  - Logistic Regression.
  - Random Forest.
  - XGBoost.
- Métricas de evaluación:
  - Accuracy.
  - Precision.
  - Recall.
  - F1 Score.
  - Robustness Score.
  - Exceso contra baseline.
- Market Signal Score integrado.
- Capa inicial de sentimiento financiero con fallback liviano.
- Backtesting vectorizado con:
  - Capital inicial simulado.
  - Benchmark Buy & Hold.
  - Comisiones.
  - Slippage.
  - Número de operaciones.
  - Costos transaccionales.
  - Exposición al mercado.
  - Sharpe Ratio aproximado.
- Validación walk-forward para series temporales.
- Visualización comparativa de modelos.
- Modularización del sistema en carpeta `src/`.
- Tests unitarios iniciales.
- Integración con GitHub Actions.
- README técnico.
- Licencia MIT.
- Archivo `.gitignore`.
- Archivo `requirements-dev.txt`.

### Limitaciones

- El proyecto no constituye asesoría financiera.
- El sentimiento financiero usa fallback liviano; FinBERT queda como extensión futura.
- El backtesting es vectorizado y no representa ejecución real de mercado.
- No se consideran impuestos, liquidez real, spreads dinámicos ni profundidad de mercado.
- La cobertura de tests inicial es limitada y debe ampliarse en futuras versiones.

### Próximas mejoras

- Integración completa con FinBERT.
- Optimización de hiperparámetros.
- Modelos adicionales: LightGBM, CatBoost, LSTM y Temporal Fusion Transformer.
- Backtesting por portafolio.
- Optimización de cartera.
- Generación automática de reportes PDF.
- API con FastAPI.
- Sistema multilingüe.
- Mayor cobertura de pruebas automatizadas.
