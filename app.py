import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import importlib

from src.indicators import add_technical_indicators
from src.data_loader import cargar_datos
from src.features import create_ml_dataset, get_feature_columns
from src import model as market_model
from src.signals import compute_market_signal
from src.financial_metrics import calculate_financial_kpis
from src.backtesting import run_backtest, calculate_backtest_metrics

market_model = importlib.reload(market_model)

st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")

st.title("Market Intelligence Dashboard")
st.markdown(
    "Análisis financiero con indicadores técnicos, Machine Learning y datos actualizados de mercado"
)

st.sidebar.header("Configuración")

activos = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "EUR/USD": "EURUSD=X"
}

seleccion = st.sidebar.selectbox("Selecciona un activo", list(activos.keys()))
periodo = st.sidebar.selectbox("Periodo", ["1y", "2y", "5y"])
ticker = activos[seleccion]


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

financial_kpis = calculate_financial_kpis(precio)


# =========================
# Dataset y modelo ML global
# =========================

df_ml = create_ml_dataset(
    precio=precio,
    ma20=ma20,
    ma50=ma50,
    rsi=rsi,
    std20=std20
)

df_backtest = run_backtest(df_ml)
backtest_metrics = calculate_backtest_metrics(df_backtest)

# Protección: crear target direccional si no viene desde features.py
if "target_direction" not in df_ml.columns:
    df_ml["target_direction"] = (df_ml["target"] > df_ml["precio"]).astype(int)

model_available = len(df_ml) >= 80

if model_available:
    feature_cols = get_feature_columns()

    X = df_ml[feature_cols]
    y = df_ml["target"]

    results = market_model.train_and_evaluate_model(X, y)
    
    y_direction = df_ml["target_direction"]

    direction_model_available = hasattr(
    market_model,
    "train_and_evaluate_direction_model"
    )

    if direction_model_available:
        direction_results = market_model.train_and_evaluate_direction_model(X, y_direction)
        direction_metrics = direction_results["direction_metrics"]
        direction_probabilities = direction_metrics["direction_probabilities"]
        latest_model_probability = float(direction_probabilities[-1])
    else:
        direction_metrics = None

    X_test = results["X_test"]
    y_test = results["y_test"]
    metrics = results["metrics"]

    preds = metrics["predictions"]
    naive_preds = metrics["naive_predictions"]

    mae_modelo = metrics["mae_model"]
    mae_naive = metrics["mae_naive"]
    rmse_modelo = metrics["rmse_model"]
    mape_modelo = metrics["mape_model"]
    mejora_vs_naive = metrics["improvement_vs_naive"]
    directional_accuracy = metrics["directional_accuracy"]


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Precio",
    "Indicadores",
    "Predicción ML",
    "Señal de Mercado",
    "Backtesting"
])


with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(precio.index),
        y=list(precio.values),
        name="Precio",
        line=dict(color="royalblue")
    ))

    fig.add_trace(go.Scatter(
        x=list(ma20.index),
        y=list(ma20.values),
        name="MA20",
        line=dict(color="orange", dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=list(ma50.index),
        y=list(ma50.values),
        name="MA50",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{seleccion} - Precio histórico"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Precio actual", f"${precio.values[-1]:.2f}")
    col2.metric("Máximo", f"${precio.values.max():.2f}")
    col3.metric("Mínimo", f"${precio.values.min():.2f}")

    col4, col5, col6 = st.columns(3)

    col4.metric(
    "Retorno período",
    f"{financial_kpis['cumulative_return']:.2f}%"
    )

    col5.metric(
    "Volatilidad anualizada",
    f"{financial_kpis['annualized_volatility']:.2f}%"
    )

    col6.metric(
    "Drawdown máximo",
    f"{financial_kpis['max_drawdown']:.2f}%"
    )


with tab2:
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=list(precio.index),
        y=list(banda_sup.values),
        name="Banda Superior",
        line=dict(color="red", dash="dash")
    ))

    fig2.add_trace(go.Scatter(
        x=list(precio.index),
        y=list(ma20.values),
        name="MA20",
        line=dict(color="orange")
    ))

    fig2.add_trace(go.Scatter(
        x=list(precio.index),
        y=list(banda_inf.values),
        name="Banda Inferior",
        line=dict(color="green", dash="dash"),
        fill="tonexty",
        fillcolor="rgba(0,255,0,0.05)"
    ))

    fig2.add_trace(go.Scatter(
        x=list(precio.index),
        y=list(precio.values),
        name="Precio",
        line=dict(color="royalblue")
    ))

    fig2.update_layout(
        template="plotly_dark",
        title="Bandas de Bollinger"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=list(rsi.index),
        y=list(rsi.values),
        name="RSI",
        line=dict(color="cyan")
    ))

    fig3.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        annotation_text="Sobrecomprado"
    )

    fig3.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        annotation_text="Sobrevendido"
    )

    fig3.update_layout(
        template="plotly_dark",
        title="RSI 14 días",
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig3, use_container_width=True)


with tab3:
    st.subheader("Modelo predictivo con validación temporal")

    if not model_available:
        st.warning("No hay suficientes datos para entrenar un modelo robusto.")
    else:
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

        st.markdown("### Modelo direccional")

        if direction_metrics is None:
            st.warning(
                "El modelo direccional aún no está disponible. "
                "Verifica que `train_and_evaluate_direction_model` exista en `src/model.py`."
            )
        else:
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)

            dcol1.metric(
                "Accuracy direccional",
                f"{direction_metrics['direction_accuracy']:.2f}%"
            )

            dcol2.metric(
                "Precision",
                f"{direction_metrics['direction_precision']:.2f}%"
            )

            dcol3.metric(
                "Recall",
                f"{direction_metrics['direction_recall']:.2f}%"
            )

            dcol4.metric(
                "F1 Score",
                f"{direction_metrics['direction_f1']:.2f}%"
            )

            st.metric(
                "Baseline direccional",
                f"{direction_metrics['direction_baseline_accuracy']:.2f}%"
            )

            if direction_metrics["improvement_vs_direction_baseline"] > 0:
                st.success(
                    f"El modelo direccional supera al baseline en "
                    f"{direction_metrics['improvement_vs_direction_baseline']:.2f} puntos porcentuales."
                )
            else:
                st.warning(
                    f"El modelo direccional NO supera al baseline. Diferencia: "
                    f"{direction_metrics['improvement_vs_direction_baseline']:.2f} puntos porcentuales."
                )


with tab4:
    st.subheader("Señal ejecutiva de mercado")

    if not model_available:
        st.warning("No hay suficientes datos para calcular una señal robusta.")
    else:
        last_price = float(precio.iloc[-1])
        last_rsi = float(rsi.iloc[-1])
        last_ma20 = float(ma20.iloc[-1])
        last_ma50 = float(ma50.iloc[-1])
        last_volatility = float(std20.iloc[-1] / last_price)

        signal_result = compute_market_signal(
            last_price=last_price,
            rsi=last_rsi,
            ma20=last_ma20,
            ma50=last_ma50,
            volatility=last_volatility,
            
        model_probability=latest_model_probability,
            sentiment_score=50.0
        )

        st.metric(
            "Market Signal Score",
        
        f"{signal_result['market_signal_score']}/100"
        )
        
        st.metric("Señal", 
        signal_result["signal"])
        
        st.write(signal_result["interpretation"])

        col1, col2, col3 = st.columns(3)

        col1, col2 = st.columns(2)
        
        col1.metric(
            "Probabilidad alcista ML",
            
        f"{signal_result['model_probability_score']}%"
        )
        
        col2.metric(
            "Score técnico",
            f"{signal_result['technical_score']}%"
        )
        col3, col4 = st.columns(2)
        
        col3.metric(
            "Score sentimiento",
            f"{signal_result['sentiment_score']}%"
        )

        col4.metric(
            "Score volatilidad",
            f"{signal_result['volatility_score']}%"
        )

        st.markdown("### Variables usadas")

        st.json({
            "Precio actual": round(last_price, 2),
            "RSI": round(last_rsi, 2),
            "MA20": round(last_ma20, 2),
            "MA50": round(last_ma50, 2),
            "Volatilidad relativa": round(last_volatility, 4),
            "Probabilidad alcista ML": round(latest_model_probability, 4),
            "Sentimiento": "Neutral temporal hasta integrar FinBERT",
        })

        st.caption(
            "El Market Signal Score integra probabilidad alcista del modelo direccional, "
            "score técnico, sentimiento financiero y ajuste por volatilidad. "
            "El sentimiento está temporalmente fijado en neutral hasta integrar FinBERT. "
            "No constituye recomendación financiera."
        )


with tab5:
    st.subheader("Backtesting de estrategia")

    st.caption(
        "Backtesting vectorizado simple basado en señal técnica. "
        "La señal se desplaza un período para reducir look-ahead bias. "
        "No incluye comisiones, slippage ni impuestos."
    )

    fig_bt = go.Figure()

    fig_bt.add_trace(go.Scatter(
        x=list(df_backtest.index),
        y=list(df_backtest["buy_hold_equity"]),
        name="Buy & Hold",
        line=dict(color="gray", dash="dot")
    ))

    fig_bt.add_trace(go.Scatter(
        x=list(df_backtest.index),
        y=list(df_backtest["strategy_equity"]),
        name="Estrategia",
        line=dict(color="royalblue")
    ))

    fig_bt.update_layout(
        template="plotly_dark",
        title="Estrategia vs Buy & Hold",
        xaxis_title="Fecha",
        yaxis_title="Capital simulado"
    )

    st.plotly_chart(fig_bt, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Retorno estrategia",
        f"{backtest_metrics['strategy_return']:.2f}%"
    )

    col2.metric(
        "Retorno Buy & Hold",
        f"{backtest_metrics['buy_hold_return']:.2f}%"
    )

    col3.metric(
        "Exceso retorno",
        f"{backtest_metrics['excess_return']:.2f}%"
    )

    col4.metric(
        "Drawdown estrategia",
        f"{backtest_metrics['max_drawdown_strategy']:.2f}%"
    )

    st.metric(
        "Exposición al mercado",
        f"{backtest_metrics['exposure']:.2f}%"
    )

    if backtest_metrics["excess_return"] > 0:
        st.success(
            "La estrategia supera a Buy & Hold en el período analizado."
        )
    else:
        st.warning(
            "La estrategia NO supera a Buy & Hold en el período analizado."
        )
