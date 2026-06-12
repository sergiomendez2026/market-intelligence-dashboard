import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide", page_icon="📈")

st.title("📈 Market Intelligence Dashboard")
st.markdown("Análisis financiero con Machine Learning en tiempo real")

# Sidebar
st.sidebar.header("⚙️ Configuración")

activos = {
    "Apple": "AAPL", "Tesla": "TSLA",
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "EUR/USD": "EURUSD=X"
}

seleccion = st.sidebar.selectbox("Selecciona un activo", list(activos.keys()))
periodo = st.sidebar.selectbox("Período", ["6mo", "1y", "2y"])
ticker = activos[seleccion]

@st.cache_data
def cargar_datos(ticker, periodo):
    df = yf.download(ticker, period=periodo, auto_adjust=True)
    return df

with st.spinner("Cargando datos..."):
    df = cargar_datos(ticker, periodo)
    precio = df["Close"].iloc[:, 0]

# Indicadores
ma20 = precio.rolling(20).mean()
ma50 = precio.rolling(50).mean()
std20 = precio.rolling(20).std()
banda_sup = ma20 + 2 * std20
banda_inf = ma20 - 2 * std20

delta = precio.diff()
ganancia = delta.where(delta > 0, 0).rolling(14).mean()
perdida = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + ganancia / perdida))

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Precio", "📉 Indicadores", "🤖 Predicción ML"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=precio.index, y=precio, name="Precio", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=precio.index, y=ma20, name="MA20", line=dict(color="orange", dash="dash")))
    fig.add_trace(go.Scatter(x=precio.index, y=ma50, name="MA50", line=dict(color="red", dash="dash")))
    fig.update_layout(template="plotly_dark", title=f"{seleccion} - Precio histórico")
    st.plotly_chart(fig, use_container_width=True)

    precio_val = float(precio.iloc[-1])
    precio_max = float(precio.max())
    precio_min = float(precio.min())
    col1.metric("Precio actual", f"${precio_val:.2f}")
    col2.metric("Máximo 52s", f"${precio_max:.2f}")
    col3.metric("Mínimo 52s", f"${precio_min:.2f}")

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=precio.index, y=banda_sup, name="Banda Superior", line=dict(color="red", dash="dash")))
    fig2.add_trace(go.Scatter(x=precio.index, y=ma20, name="MA20", line=dict(color="orange")))
    fig2.add_trace(go.Scatter(x=precio.index, y=banda_inf, name="Banda Inferior", line=dict(color="green", dash="dash"), fill="tonexty", fillcolor="rgba(0,255,0,0.05)"))
    fig2.add_trace(go.Scatter(x=precio.index, y=precio, name="Precio", line=dict(color="royalblue")))
    fig2.update_layout(template="plotly_dark", title="Bandas de Bollinger")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="cyan")))
    fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado")
    fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido")
    fig3.update_layout(template="plotly_dark", title="RSI (14 días)", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    df_ml = pd.DataFrame()
    df_ml["precio"] = precio
    df_ml["ma20"] = ma20
    df_ml["ma50"] = ma50
    df_ml["rsi"] = rsi
    df_ml["std20"] = std20
    df_ml["retorno_1d"] = precio.pct_change(1)
    df_ml["retorno_5d"] = precio.pct_change(5)
    df_ml["target"] = precio.shift(-1)
    df_ml = df_ml.dropna()

    X = df_ml.drop("target", axis=1)
    y = df_ml["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    modelo = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name="Real", line=dict(color="royalblue")))
    fig4.add_trace(go.Scatter(x=y_test.index, y=preds, name="Predicción", line=dict(color="orange", dash="dash")))
    fig4.update_layout(template="plotly_dark", title="XGBoost: Predicción vs Precio Real")
    st.plotly_chart(fig4, use_container_width=True)

    st.metric("Error promedio del modelo (MAE)", f"${mae:.2f}")
