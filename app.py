import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")

st.title("Market Intelligence Dashboard")
st.markdown("Analisis financiero con Machine Learning en tiempo real")

st.sidebar.header("Configuracion")

activos = {
    "Apple": "AAPL", "Tesla": "TSLA",
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "EUR/USD": "EURUSD=X"
}

seleccion = st.sidebar.selectbox("Selecciona un activo", list(activos.keys()))
periodo = st.sidebar.selectbox("Periodo", ["6mo", "1y", "2y"])
ticker = activos[seleccion]

@st.cache_data
def cargar_datos(ticker, periodo):
    df = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
    close = df["Close"]
    # Aplanar cualquier estructura MultiIndex
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    arr = np.array(close).flatten().astype(float)
    serie = pd.Series(arr, index=df.index[-len(arr):], name="Close")
    return serie.dropna()

with st.spinner("Cargando datos..."):
    precio = cargar_datos(ticker, periodo)

if len(precio) < 60:
    st.error("No hay suficientes datos. Selecciona otro periodo.")
    st.stop()

ma20 = precio.rolling(20).mean()
ma50 = precio.rolling(50).mean()
std20 = precio.rolling(20).std()
banda_sup = ma20 + 2 * std20
banda_inf = ma20 - 2 * std20

delta = precio.diff()
ganancia = delta.where(delta > 0, 0).rolling(14).mean()
perdida = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + ganancia / perdida))

tab1, tab2, tab3 = st.tabs(["Precio", "Indicadores", "Prediccion ML"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(precio.index), y=list(precio.values), name="Precio", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=list(ma20.index), y=list(ma20.values), name="MA20", line=dict(color="orange", dash="dash")))
    fig.add_trace(go.Scatter(x=list(ma50.index), y=list(ma50.values), name="MA50", line=dict(color="red", dash="dash")))
    fig.update_layout(template="plotly_dark", title=f"{seleccion} - Precio historico")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precio actual", f"${precio.values[-1]:.2f}")
    col2.metric("Maximo", f"${precio.values.max():.2f}")
    col3.metric("Minimo", f"${precio.values.min():.2f}")

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(precio.index), y=list(banda_sup.values), name="Banda Superior", line=dict(color="red", dash="dash")))
    fig2.add_trace(go.Scatter(x=list(precio.index), y=list(ma20.values), name="MA20", line=dict(color="orange")))
    fig2.add_trace(go.Scatter(x=list(precio.index), y=list(banda_inf.values), name="Banda Inferior", line=dict(color="green", dash="dash"), fill="tonexty", fillcolor="rgba(0,255,0,0.05)"))
    fig2.add_trace(go.Scatter(x=list(precio.index), y=list(precio.values), name="Precio", line=dict(color="royalblue")))
    fig2.update_layout(template="plotly_dark", title="Bandas de Bollinger")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(rsi.index), y=list(rsi.values), name="RSI", line=dict(color="cyan")))
    fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado")
    fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido")
    fig3.update_layout(template="plotly_dark", title="RSI 14 dias", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    p = precio.values.astype(float)
    idx = precio.index

    ma20v = ma20.values.astype(float)
    ma50v = ma50.values.astype(float)
    rsiv = rsi.values.astype(float)
    std20v = std20.values.astype(float)
    ret1 = pd.Series(p).pct_change(1).values
    ret5 = pd.Series(p).pct_change(5).values
    target = np.roll(p, -1)

    df_ml = pd.DataFrame({
        "precio": p, "ma20": ma20v, "ma50": ma50v,
        "rsi": rsiv, "std20": std20v,
        "retorno_1d": ret1, "retorno_5d": ret5,
        "target": target
    }, index=idx)

    df_ml = df_ml.iloc[:-1].dropna()

    X = df_ml.drop("target", axis=1)
    y = df_ml["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    modelo = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=list(y_test.index), y=list(y_test.values), name="Real", line=dict(color="royalblue")))
    fig4.add_trace(go.Scatter(x=list(y_test.index), y=list(preds), name="Prediccion", line=dict(color="orange", dash="dash")))
    fig4.update_layout(template="plotly_dark", title="XGBoost: Prediccion vs Precio Real")
    st.plotly_chart(fig4, use_container_width=True)

    st.metric("Error promedio del modelo (MAE)", f"${mae:.2f}")
