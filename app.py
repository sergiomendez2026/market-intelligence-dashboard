import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.indicators import add_technical_indicators
from src.data_loader import cargar_datos
from src.features import create_ml_dataset, get_feature_columns
from src import model as market_model
from src import signals as market_signals
from src.financial_metrics import calculate_financial_kpis
from src.backtesting import run_backtest, calculate_backtest_metrics
from src.sentiment import calculate_fallback_sentiment_score
from src.ui import (
    render_header,
    render_disclaimer,
    render_sidebar_assets,
    render_model_warning,
    render_methodology_note
)
from src.validation import (
    walk_forward_regression_validation,
    walk_forward_direction_validation
)
from src.model_comparison import compare_direction_models
from src.portfolio import (
    build_price_matrix,
    calculate_portfolio_metrics,
    equal_weight_vector
)
from src.baselines import compare_academic_baselines
from src.walkforward_academic import (
    compare_walkforward_academic_models,
    collect_walkforward_predictions_for_statistics
)
from src.statistical_tests import compare_model_predictions_statistically



@st.cache_data(ttl=3600)
def load_market_data_cached(ticker: str, periodo: str):
    return cargar_datos(ticker, periodo)


@st.cache_data(ttl=3600)
def calculate_indicators_cached(precio: pd.Series):
    df_price = pd.DataFrame({"Close": precio})
    return add_technical_indicators(df_price)


@st.cache_data(ttl=3600)
def create_ml_dataset_cached(
    precio: pd.Series,
    ma20: pd.Series,
    ma50: pd.Series,
    rsi: pd.Series,
    std20: pd.Series,
    feature_version: str = "features_v2"
):
    return create_ml_dataset(
        precio=precio,
        ma20=ma20,
        ma50=ma50,
        rsi=rsi,
        std20=std20
    )


@st.cache_data(ttl=3600)
def calculate_financial_kpis_cached(precio: pd.Series):
    return calculate_financial_kpis(precio)


@st.cache_data(ttl=3600)
def run_backtest_cached(
    df_ml: pd.DataFrame,
    commission: float,
    slippage: float
):
    df_backtest = run_backtest(
        df_ml,
        commission=commission,
        slippage=slippage
    )
    
    backtest_metrics = calculate_backtest_metrics(df_backtest)
    
    return df_backtest, backtest_metrics

@st.cache_data(ttl=3600)
def train_regression_model_cached(X: pd.DataFrame, y: pd.Series):
    return market_model.train_and_evaluate_model(X, y)


@st.cache_data(ttl=3600)
def train_direction_model_cached(X: pd.DataFrame, y_direction: pd.Series):
    return market_model.train_and_evaluate_direction_model(X, y_direction)

st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")

@st.cache_data(ttl=3600)
def build_price_matrix_cached(
    selected_assets: list[str],
    asset_tickers: dict[str, str],
    periodo: str
):
    return build_price_matrix(
        selected_assets=selected_assets,
        asset_tickers=asset_tickers,
        periodo=periodo,
        data_loader_func=load_market_data_cached
    )

render_header()
render_disclaimer()

seleccion, ticker, periodo = render_sidebar_assets()
asset_tickers = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "EUR/USD": "EURUSD=X"
}

st.sidebar.markdown("---")
st.sidebar.subheader("Supuestos de backtesting")

commission = st.sidebar.number_input(
    "Comisión por operación (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.05
) / 100

slippage = st.sidebar.number_input(
    "Slippage por operación (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.05
) / 100

run_walk_forward = st.sidebar.checkbox(
    "Ejecutar validación walk-forward",
    value=False
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Costos de backtesting")

commission = st.sidebar.number_input(
    "Comisión por operación (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.01
) / 100

slippage = st.sidebar.number_input(
    "Slippage estimado (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.05,
    step=0.01
) / 100


with st.spinner("Cargando datos de mercado..."):
    precio = load_market_data_cached(ticker, periodo)

if len(precio) < 60:
    st.error("No hay suficientes datos. Selecciona otro periodo.")
    st.stop()


df_indicators = calculate_indicators_cached(precio)

precio = df_indicators["Close"]
ma20 = df_indicators["MA20"]
ma50 = df_indicators["MA50"]
banda_sup = df_indicators["BB_Upper"]
banda_inf = df_indicators["BB_Lower"]
std20 = df_indicators["Close"].rolling(20).std()
rsi = df_indicators["RSI"]

financial_kpis = calculate_financial_kpis_cached(precio)
sample_news = [
    f"{seleccion} financial market outlook",
    f"{seleccion} earnings growth risk outlook",
    f"{seleccion} analyst sentiment market performance"
]

sentiment_result = calculate_fallback_sentiment_score(sample_news)
sentiment_score = sentiment_result["sentiment_score"]

# =========================
# Dataset y modelo ML global
# =========================

df_ml = create_ml_dataset_cached(
    precio=precio,
    ma20=ma20,
    ma50=ma50,
    rsi=rsi,
    std20=std20
)

df_backtest, backtest_metrics = run_backtest_cached(
    df_ml,
    commission,
    slippage
)

# Protección: crear target direccional si no viene desde features.py
if "target_direction" not in df_ml.columns:
    df_ml["target_direction"] = (df_ml["target"] > df_ml["precio"]).astype(int)

model_available = len(df_ml) >= 80

latest_model_probability = 0.50
direction_metrics = None

if model_available:
    feature_cols = get_feature_columns()

    missing_features = [col for col in feature_cols if col not in df_ml.columns]

    if missing_features:
        st.error(
            "Error de columnas en el dataset ML. "
            "Estas variables están en get_feature_columns(), "
            "pero no existen en df_ml:"
        )
        st.write(missing_features)
        st.write("Columnas disponibles en df_ml:")
        st.write(list(df_ml.columns))
        st.stop()
        
    X = df_ml[feature_cols]
    y = df_ml["target"]

    results = train_regression_model_cached(X, y)
    
    y_direction = df_ml["target_direction"]

    model_comparison_df = compare_direction_models(X, y_direction)

    direction_model_available = hasattr(
    market_model,
    "train_and_evaluate_direction_model"
    )

    if direction_model_available:
        direction_results = train_direction_model_cached(X, y_direction)
        direction_metrics = direction_results["direction_metrics"]
        direction_probabilities = direction_metrics["direction_probabilities"]
        latest_model_probability = float(direction_probabilities[-1])
    else:
        direction_metrics = None
        latest_model_probability = 0.50

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
    if run_walk_forward:
        wf_regression = walk_forward_regression_validation(
            X=X,
            y=y,
            min_train_size=120,
            test_size=20,
            step_size=20
        )

        wf_direction = walk_forward_direction_validation(
            X=X,
            y_direction=y_direction,
            min_train_size=120,
            test_size=20,
            step_size=20
        )
    else:
        wf_regression = None
        wf_direction = None

else:
    wf_regression = None
    wf_direction = None

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Precio",
    "Indicadores",
    "Predicción ML",
    "Señal de Mercado",
    "Backtesting",
    "Validación",
    "Modelos",
    "Portafolio",
    "Baselines Académicos",
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
    render_methodology_note()

    if not model_available:
        render_model_warning()
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

        signal_result = market_signals.compute_market_signal(
            last_price=last_price,
            rsi=last_rsi,
            ma20=last_ma20,
            ma50=last_ma50,
            volatility=last_volatility,
            model_probability=latest_model_probability,
            sentiment_score=sentiment_score
        )

        st.metric(
            "Market Signal Score",
            f"{signal_result['market_signal_score']}/100"
        )

        st.metric(
            "Señal",
            signal_result["signal"]
        )

        st.write(signal_result["interpretation"])

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
            "Sentimiento": sentiment_result["sentiment_label"],
            "Método sentimiento": sentiment_result["method"],
            "Positive hits": sentiment_result["positive_hits"],
            "Negative hits": sentiment_result["negative_hits"]
        })

        st.caption(
            "El Market Signal Score integra probabilidad alcista del modelo direccional, "
            "score técnico, sentimiento financiero y ajuste por volatilidad. "
            "El sentimiento usa un fallback liviano basado en términos financieros; FinBERT puede integrarse como capa avanzada. "
            "No constituye recomendación financiera."
        )


with tab5:
    st.subheader("Backtesting de estrategia")

    st.caption(
        "Backtesting vectorizado basado en señal técnica. "
        "La señal se desplaza un período para reducir look-ahead bias. "
        "Incluye comisión y slippage estimados. No incluye impuestos ni impacto de liquidez."
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
    
    col5, col6, col7, col8 = st.columns(4)
    
    col5.metric(
        "Drawdown Buy & Hold",
        f"{backtest_metrics['max_drawdown_buy_hold']:.2f}%"
    )
    
    col6.metric(
        "Exposición al mercado",
        f"{backtest_metrics['exposure']:.2f}%"
    )
    
    col7.metric(
        "Número de operaciones",
        f"{backtest_metrics['number_of_trades']}"
    )
    
    col8.metric(
        "Sharpe estrategia",
        f"{backtest_metrics['sharpe_ratio']:.2f}"
    )
    
    col9, col10 = st.columns(2)
    
    col9.metric(
        "Turnover promedio",
        f"{backtest_metrics['turnover']:.2f}%"
    )
    
    col10.metric(
        "Costo transaccional estimado",
        f"{backtest_metrics['total_transaction_cost']:.2f}%"
    )
    
    if backtest_metrics["excess_return"] > 0:
        st.success(
            "La estrategia supera a Buy & Hold en el período analizado."
        )
    else:
        st.warning(
            "La estrategia NO supera a Buy & Hold en el período analizado."
        )

    st.caption(
        f"Supuestos del backtest: comisión {commission * 100:.2f}% por operación, "
        f"slippage {slippage * 100:.2f}% por operación. "
        "La señal se desplaza un período para reducir look-ahead bias. "
        "El backtest no garantiza rentabilidad futura."
    )

with tab6:
    st.subheader("Validación walk-forward")

    st.caption(
        "La validación walk-forward entrena el modelo usando solo datos pasados "
        "y evalúa en bloques futuros. Este método es más riguroso que una única "
        "división 80/20 porque simula mejor el uso real en series temporales."
    )

    if not model_available:
        st.warning("No hay suficientes datos para ejecutar validación walk-forward.")

    elif not run_walk_forward:
        st.info(
            "La validación walk-forward está desactivada para mejorar el tiempo de carga. "
            "Actívala desde el panel lateral cuando quieras ejecutarla."
        )

    else:
        st.markdown("### Regresión de precio")

        if not wf_regression["available"]:
            st.warning(wf_regression["message"])
        else:
            wf_reg_metrics = wf_regression["metrics"]
            wf_reg_results = wf_regression["results"]

            fig_wf_reg = go.Figure()

            fig_wf_reg.add_trace(go.Scatter(
                x=list(wf_reg_results.index),
                y=list(wf_reg_results["actual"]),
                name="Precio real",
                line=dict(color="royalblue")
            ))

            fig_wf_reg.add_trace(go.Scatter(
                x=list(wf_reg_results.index),
                y=list(wf_reg_results["prediction"]),
                name="Predicción walk-forward",
                line=dict(color="orange", dash="dash")
            ))

            fig_wf_reg.add_trace(go.Scatter(
                x=list(wf_reg_results.index),
                y=list(wf_reg_results["naive_prediction"]),
                name="Baseline naïve",
                line=dict(color="gray", dash="dot")
            ))

            fig_wf_reg.update_layout(
                template="plotly_dark",
                title="Walk-forward: Predicción vs Real vs Baseline",
                xaxis_title="Fecha",
                yaxis_title="Precio"
            )

            st.plotly_chart(fig_wf_reg, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "MAE walk-forward",
                f"{wf_reg_metrics['mae_model']:.2f}"
            )

            col2.metric(
                "MAE baseline",
                f"{wf_reg_metrics['mae_naive']:.2f}"
            )

            col3.metric(
                "MAPE",
                f"{wf_reg_metrics['mape_model']:.2f}%"
            )

            col4.metric(
                "Dirección correcta",
                f"{wf_reg_metrics['directional_accuracy']:.2f}%"
            )

            st.metric(
                "Observaciones evaluadas",
                f"{wf_reg_metrics['n_predictions']}"
            )

            if wf_reg_metrics["improvement_vs_naive"] > 0:
                st.success(
                    f"El modelo walk-forward supera al baseline naïve en "
                    f"{wf_reg_metrics['improvement_vs_naive']:.2f}%."
                )
            else:
                st.warning(
                    f"El modelo walk-forward NO supera al baseline naïve. "
                    f"Diferencia: {wf_reg_metrics['improvement_vs_naive']:.2f}%."
                )

        st.markdown("---")
        st.markdown("### Clasificación direccional")

        if not wf_direction["available"]:
            st.warning(wf_direction["message"])
        else:
            wf_dir_metrics = wf_direction["metrics"]
            wf_dir_results = wf_direction["results"]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{wf_dir_metrics['direction_accuracy']:.2f}%"
            )

            col2.metric(
                "Precision",
                f"{wf_dir_metrics['direction_precision']:.2f}%"
            )

            col3.metric(
                "Recall",
                f"{wf_dir_metrics['direction_recall']:.2f}%"
            )

            col4.metric(
                "F1 Score",
                f"{wf_dir_metrics['direction_f1']:.2f}%"
            )

            st.metric(
                "Baseline direccional",
                f"{wf_dir_metrics['direction_baseline_accuracy']:.2f}%"
            )

            st.metric(
                "Observaciones evaluadas",
                f"{wf_dir_metrics['n_predictions']}"
            )

            if wf_dir_metrics["improvement_vs_direction_baseline"] > 0:
                st.success(
                    f"El modelo direccional walk-forward supera al baseline en "
                    f"{wf_dir_metrics['improvement_vs_direction_baseline']:.2f} puntos porcentuales."
                )
            else:
                st.warning(
                    f"El modelo direccional walk-forward NO supera al baseline. "
                    f"Diferencia: {wf_dir_metrics['improvement_vs_direction_baseline']:.2f} puntos porcentuales."
                )

            st.dataframe(
                wf_dir_results.tail(20),
                use_container_width=True
            )
            
with tab7:
    st.subheader("Comparación de modelos direccionales")

    st.caption(
        "Esta sección compara modelos supervisados para predicción direccional. "
        "El objetivo no es prometer rentabilidad, sino evaluar si algún modelo supera "
        "un baseline simple bajo validación temporal."
    )

    if not model_available:
        st.warning("No hay suficientes datos para comparar modelos.")
    else:
        st.dataframe(
            model_comparison_df,
            use_container_width=True
        )
        st.markdown("### Comparación visual de desempeño")

        fig_models = go.Figure()

        fig_models.add_trace(go.Bar(
            x=model_comparison_df["Modelo"],
            y=model_comparison_df["Accuracy"],
            name="Accuracy"
        ))
        
        fig_models.add_trace(go.Bar(
            x=model_comparison_df["Modelo"],
            y=model_comparison_df["F1 Score"],
            name="F1 Score"
        ))
        
        fig_models.add_trace(go.Bar(
            x=model_comparison_df["Modelo"],
            y=model_comparison_df["Robustness Score"],
            name="Robustness Score"
        ))
        
        fig_models.update_layout(
            template="plotly_dark",
            title="Comparación de modelos direccionales",
            xaxis_title="Modelo",
            yaxis_title="Score (%)",
            barmode="group",
            yaxis=dict(range=[0, 100]),
            legend_title="Métrica"
        )
        
        st.plotly_chart(fig_models, use_container_width=True)

        st.markdown("### Exceso de accuracy contra baseline")
        
        model_only_df = model_comparison_df[
            model_comparison_df["Modelo"] != "Baseline Dummy"
        ].copy()
        
        fig_excess = go.Figure()
        
        fig_excess.add_trace(go.Bar(
            x=model_only_df["Modelo"],
            y=model_only_df["Exceso Accuracy vs Baseline"],
            name="Exceso vs baseline"
        ))
        
        fig_excess.add_hline(
            y=0,
            line_dash="dash",
            annotation_text="Baseline",
            annotation_position="top left"
        )
        
        fig_excess.update_layout(
            template="plotly_dark",
            title="Valor incremental de cada modelo frente al baseline",
            xaxis_title="Modelo",
            yaxis_title="Exceso de accuracy (puntos porcentuales)",
            legend_title="Métrica"
        )
        
        st.plotly_chart(fig_excess, use_container_width=True)

        candidate_models = model_comparison_df[
            model_comparison_df["Modelo"] != "Baseline Dummy"
        ].copy()

        best_model_row = candidate_models.sort_values(
            by="Robustness Score",
            ascending=False
        ).iloc[0]
        
        st.markdown("### Mejor modelo productivo según Robustness Score")

        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Modelo", best_model_row["Modelo"])
        col2.metric("Accuracy", f"{best_model_row['Accuracy']:.2f}%")
        col3.metric("F1 Score", f"{best_model_row['F1 Score']:.2f}%")
        col4.metric("Robustness Score", f"{best_model_row['Robustness Score']:.2f}")

        st.metric(
            "Exceso Accuracy vs baseline",
            f"{best_model_row['Exceso Accuracy vs Baseline']:.2f} pp"
        )

        if best_model_row["Exceso Accuracy vs Baseline"] > 0:
            st.success(
                "Existe al menos un modelo que supera al baseline direccional "
                "en el período evaluado."
            )
        else:
            st.warning(
                "Ningún modelo supera claramente al baseline direccional. "
                "Esto es una señal metodológica sana: el sistema evita sobreprometer."
            )

        st.caption(
            "Nota metodológica: en mercados financieros, superar un baseline simple "
            "de forma estable es difícil. Por eso se reportan modelos alternativos, "
            "baseline y métricas de clasificación."
        )

with tab8:
    st.subheader("Analítica de portafolio")

    st.caption(
        "Esta sección permite evaluar un portafolio equiponderado de múltiples activos. "
        "El objetivo es analizar retorno, volatilidad, drawdown, Sharpe Ratio y correlación. "
        "No constituye recomendación de inversión."
    )

    selected_portfolio_assets = st.multiselect(
        "Selecciona activos para el portafolio",
        list(asset_tickers.keys()),
        default=["Apple", "Tesla", "Bitcoin", "S&P 500"]
    )

    if len(selected_portfolio_assets) < 2:
        st.warning("Selecciona al menos dos activos para construir un portafolio.")
    else:
        prices_portfolio = build_price_matrix_cached(
            selected_assets=selected_portfolio_assets,
            asset_tickers=asset_tickers,
            periodo=periodo
        )

        weights = equal_weight_vector(len(selected_portfolio_assets))

        portfolio_result = calculate_portfolio_metrics(
            prices_df=prices_portfolio,
            weights=weights
        )

        if not portfolio_result["available"]:
            st.warning(portfolio_result["message"])
        else:
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Retorno acumulado",
                f"{portfolio_result['cumulative_return']:.2f}%"
            )

            col2.metric(
                "Retorno anualizado",
                f"{portfolio_result['annualized_return']:.2f}%"
            )

            col3.metric(
                "Volatilidad anualizada",
                f"{portfolio_result['annualized_volatility']:.2f}%"
            )

            col4.metric(
                "Sharpe Ratio",
                f"{portfolio_result['sharpe_ratio']:.2f}"
            )

            st.metric(
                "Drawdown máximo",
                f"{portfolio_result['max_drawdown']:.2f}%"
            )

            st.markdown("### Curva de capital del portafolio")

            fig_portfolio = go.Figure()

            fig_portfolio.add_trace(go.Scatter(
                x=list(portfolio_result["equity_curve"].index),
                y=list(portfolio_result["equity_curve"].values),
                name="Portafolio equiponderado",
                line=dict(color="royalblue")
            ))

            fig_portfolio.update_layout(
                template="plotly_dark",
                title="Evolución del capital simulado",
                xaxis_title="Fecha",
                yaxis_title="Capital simulado"
            )

            st.plotly_chart(fig_portfolio, use_container_width=True)

            st.markdown("### Matriz de correlación")

            corr = portfolio_result["correlation_matrix"]

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlación")
            ))

            fig_corr.update_layout(
                template="plotly_dark",
                title="Correlación entre activos"
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("### Pesos del portafolio")

            weights_df = pd.DataFrame({
                "Activo": selected_portfolio_assets,
                "Peso": weights
            })

            weights_df["Peso"] = weights_df["Peso"].apply(
                lambda x: f"{x * 100:.2f}%"
            )

            st.dataframe(weights_df, use_container_width=True)

            st.caption(
                "El portafolio actual usa pesos iguales. "
                "En una siguiente versión se podrán configurar pesos personalizados "
                "y aplicar optimización de portafolio."
            )
            
with tab9:
    st.subheader("Baselines académicos")

    st.caption(
        "Esta sección compara modelos baseline obligatorios para investigación académica: "
        "Naive t+1 = precio actual, regresión lineal y ARIMA. "
        "Estos modelos permiten evaluar si los enfoques más complejos realmente agregan valor."
    )

    try:
        price_series = precio.dropna()

        baseline_results = compare_academic_baselines(
            prices=price_series,
        )

        st.dataframe(baseline_results, use_container_width=True)

        st.markdown("### Interpretación")

        st.info(
            "El modelo Naive es el punto de comparación mínimo. "
            "Si un modelo complejo no supera al Naive bajo validación temporal, "
            "no se debe interpretar como superior aunque use machine learning avanzado."
        )

        st.markdown(
            """
            **Lectura académica recomendada:**

            - Si ARIMA supera al modelo ML, el enfoque econométrico simple puede ser suficiente.
            - Si XGBoost supera a Naive y ARIMA, existe evidencia inicial de valor predictivo no lineal.
            - Si el modelo técnico + FinBERT supera al modelo técnico sin sentimiento, entonces el sentimiento financiero podría aportar información incremental.
            - La mejora debe validarse con walk-forward y prueba estadística, no solo con una tabla puntual.
            """
        )

        st.markdown("---")
        st.subheader("Validación walk-forward comparativa")

        st.caption(
            "Esta sección evalúa los modelos mediante múltiples ventanas temporales. "
            "Esto es más riguroso que un único split 80/20 porque simula un escenario "
            "donde el modelo entrena con datos pasados y predice datos futuros."
        )

        run_academic_walkforward = st.checkbox(
            "Ejecutar walk-forward académico",
            value=False,
            help="Puede tardar más porque ARIMA se reentrena por ventanas.",
        )

        if run_academic_walkforward:
            with st.spinner("Ejecutando validación walk-forward académica..."):
                try:
                    wf_results = compare_walkforward_academic_models(
                        prices=price_series,
                        initial_train_size=252,
                        test_window=20,
                        step_size=20,
                        max_windows=8,
                        include_arima=True,
                    )

                    st.dataframe(wf_results, use_container_width=True)

                    if "F1 Score" in wf_results.columns:
                        wf_plot_data = wf_results.dropna(subset=["F1 Score"])

                        fig_wf_f1 = go.Figure()

                        fig_wf_f1.add_trace(
                            go.Bar(
                                x=wf_plot_data["Modelo"],
                                y=wf_plot_data["F1 Score"],
                                name="F1 Score",
                            )
                        )

                        fig_wf_f1.update_layout(
                            template="plotly_dark",
                            title="Comparación walk-forward por F1 Score",
                            xaxis_title="Modelo",
                            yaxis_title="F1 Score (%)",
                        )

                        st.plotly_chart(fig_wf_f1, use_container_width=True)

                    st.info(
                        "La métrica principal académica es F1 Score bajo walk-forward. "
                        "Un modelo complejo debe superar al Naive y a los baselines clásicos "
                        "para considerarse metodológicamente superior."
                    )

                except Exception as error:
                    st.error(
                        f"No se pudo ejecutar la validación walk-forward académica: {error}"
                    )
        else:
            st.warning(
                "Activa la validación walk-forward académica para ejecutar una comparación más rigurosa. "
                "Se deja desactivada por defecto para no ralentizar la aplicación."
            )

    except Exception as error:
        st.error(f"No se pudieron calcular los baselines académicos: {error}")
