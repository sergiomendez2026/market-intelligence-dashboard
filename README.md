# Market Intelligence Dashboard

Aplicación web de análisis financiero con Machine Learning en tiempo real.

**Live Demo:** [Ver aplicación](https://TU-LINK.streamlit.app)

## Descripción

Dashboard interactivo que analiza activos financieros en tiempo real usando datos históricos reales. Incluye indicadores técnicos profesionales y un modelo de Machine Learning para predicción de precios.

## Activos analizados

- Acciones: Apple (AAPL), Tesla (TSLA)
- Criptomonedas: Bitcoin (BTC), Ethereum (ETH)
- Índices: S&P 500, NASDAQ
- Forex: EUR/USD

## Funcionalidades

- Precio histórico con Medias Móviles (MA20, MA50)
- Bandas de Bollinger para análisis de volatilidad
- RSI (Relative Strength Index) para señales de compra/venta
- Modelo XGBoost con MAE de ~$10 sobre datos reales
- Selector dinámico de activo y período de tiempo

## Tecnologías

- Python, Pandas, NumPy
- Plotly (visualización interactiva)
- XGBoost + Scikit-learn (Machine Learning)
- Streamlit (aplicación web)
- yFinance (datos financieros en tiempo real)
- GitHub + Streamlit Cloud (despliegue)

## Cómo ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
