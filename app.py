import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from src.indicators import add_technical_indicators

st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")

st.title("Market Intelligence Dashboard")
st.markdown("Análisis financiero con indicadores técnicos, Machine Learning y datos actualizados de mercado")

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

df_price = pd.DataFrame({"Close": precio})
df_indicators = add_technical_indicators(df_price)

precio = df_indicators["Close"]
ma20 = df_indicators["MA20"]
ma50 = df_indicators["MA50"]
banda_sup = df_indicators["BB_Upper"]
banda_inf = df_indicators["BB_Lower"]
std20 = df_indicators["Close"].rolling(20).std()
rsi = df_indicators["RSI"]

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
    st.subheader("Modelo predictivo con validación temporal")

    p = precio.values.astype(float)
    idx = precio.index

    ma20v = ma20.values.astype(float)
    ma50v = ma50.values.astype(float)
    rsiv = rsi.values.astype(float)
    std20v = std20.values.astype(float)

    ret1 = pd.Series(p, index=idx).pct_change(1)
    ret5 = pd.Series(p, index=idx).pct_change(5)

    df_ml = pd.DataFrame({
        "precio": p,
        "ma20": ma20v,
        "ma50": ma50v,
        "rsi": rsiv,
        "std20": std20v,
        "retorno_1d": ret1.values,
        "retorno_5d": ret5.values,
        "target": pd.Series(p, index=idx).shift(-1).values
    }, index=idx)

    df_ml = df_ml.dropna()

    if len(df_ml) < 80:
        st.warning("No hay suficientes datos para entrenar un modelo robusto.")
        st.stop()

    X = df_ml.drop("target", axis=1)
    y = df_ml["target"]

    split_index = int(len(df_ml) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    modelo = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)

    # Baseline naïve: mañana = hoy
    naive_preds = X_test["precio"].values

    mae_modelo = mean_absolute_error(y_test, preds)
    mae_naive = mean_absolute_error(y_test, naive_preds)

    rmse_modelo = np.sqrt(np.mean((y_test.values - preds) ** 2))
    mape_modelo = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100

    mejora_vs_naive = ((mae_naive - mae_modelo) / mae_naive) * 100

    direccion_real = np.sign(y_test.values - X_test["precio"].values)
    direccion_predicha = np.sign(preds - X_test["precio"].values)

    directional_accuracy = np.mean(direccion_real == direccion_predicha) * 100

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=list(y_test.index),
        y=list(y_test.values),
        name="Precio real",
        line=dict(color="royalblue")
    ))

    fig4.add_trace(go.Scatter(
        x=list(y_test.index),
        y=list(preds),
        name="Predicción XGBoost",
        line=dict(color="orange", dash="dash")
    ))

    fig4.add_trace(go.Scatter(
        x=list(y_test.index),
        y=list(naive_preds),
        name="Baseline naïve",
        line=dict(color="gray", dash="dot")
    ))

    fig4.update_layout(
        template="plotly_dark",
        title="XGBoost vs Precio Real vs Baseline Naïve",
        xaxis_title="Fecha",
        yaxis_title="Precio"
    )

    st.plotly_chart(fig4, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE XGBoost", f"{mae_modelo:.2f}")
    col2.metric("MAE Baseline", f"{mae_naive:.2f}")
    col3.metric("MAPE", f"{mape_modelo:.2f}%")
    col4.metric("Dirección correcta", f"{directional_accuracy:.2f}%")

    if mejora_vs_naive > 0:
        st.success(f"El modelo supera al baseline naïve en {mejora_vs_naive:.2f}%.")
    else:
        st.warning(f"El modelo NO supera al baseline naïve. Diferencia: {mejora_vs_naive:.2f}%.")

    st.caption(
        "Nota: este modelo es experimental y no constituye recomendación financiera. "
        "Evalúa patrones históricos, indicadores técnicos y comportamiento pasado del activo."
    )
