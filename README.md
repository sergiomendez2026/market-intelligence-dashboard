# Market Intelligence Dashboard

Plataforma analítica para evaluación de activos financieros mediante indicadores técnicos, machine learning, señales integradas, sentimiento financiero y backtesting.

> Proyecto académico/profesional con fines educativos y analíticos.  
> No constituye asesoría financiera, recomendación de inversión ni promesa de rentabilidad.

---

## 1. Objetivo del proyecto

El objetivo de este proyecto es construir un sistema modular de inteligencia de mercado capaz de:

- Descargar datos históricos de activos financieros.
- Calcular indicadores técnicos.
- Entrenar modelos de machine learning sobre series temporales.
- Evaluar modelos contra baselines simples.
- Generar una señal ejecutiva de mercado.
- Ejecutar backtesting con costos transaccionales.
- Aplicar validación walk-forward para reducir sobreajuste.
- Presentar resultados en una interfaz ejecutiva con Streamlit.

---

## 2. Activos soportados

Actualmente la aplicación permite analizar:

- Apple — `AAPL`
- Tesla — `TSLA`
- Bitcoin — `BTC-USD`
- Ethereum — `ETH-USD`
- S&P 500 — `^GSPC`
- NASDAQ — `^IXIC`
- EUR/USD — `EURUSD=X`

Los datos son obtenidos desde fuentes financieras públicas compatibles con `yfinance`.

---

## 3. Arquitectura del sistema

```text
market-intelligence-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
└── src/
    ├── data_loader.py
    ├── indicators.py
    ├── features.py
    ├── model.py
    ├── signals.py
    ├── sentiment.py
    ├── financial_metrics.py
    ├── backtesting.py
    ├── validation.py
    └── ui.py
