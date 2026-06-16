# src/ui.py

import streamlit as st


def render_header() -> None:
    """
    Renderiza el encabezado ejecutivo del dashboard.
    """
    st.title("Market Intelligence Dashboard")
    st.markdown(
        "Plataforma analítica para evaluación de activos financieros mediante "
        "indicadores técnicos, machine learning, señales integradas y backtesting."
    )


def render_disclaimer() -> None:
    """
    Disclaimer financiero obligatorio.
    """
    st.caption(
        "Aviso: esta aplicación tiene fines educativos y analíticos. "
        "No constituye asesoría financiera, recomendación de inversión ni promesa de rentabilidad."
    )


def render_sidebar_assets() -> tuple[str, str, str]:
    """
    Renderiza la barra lateral y devuelve selección, ticker y período.
    """
    st.sidebar.header("Configuración del análisis")

    activos = {
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "EUR/USD": "EURUSD=X"
    }

    seleccion = st.sidebar.selectbox(
        "Activo financiero",
        list(activos.keys())
    )

    periodo = st.sidebar.selectbox(
        "Horizonte histórico",
        ["1y", "2y", "5y"],
        index=1
    )

    ticker = activos[seleccion]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Interpretación")
    st.sidebar.markdown(
        """
        **1y**: último año histórico  
        **2y**: últimos dos años  
        **5y**: últimos cinco años  
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Estado del producto")
    st.sidebar.success("Versión académica/profesional en desarrollo")

    return seleccion, ticker, periodo


def render_metric_card(label: str, value: str, help_text: str | None = None) -> None:
    """
    Wrapper simple para métricas ejecutivas.
    """
    st.metric(label=label, value=value, help=help_text)


def render_model_warning() -> None:
    """
    Mensaje estándar cuando no hay suficientes datos.
    """
    st.warning(
        "No hay suficientes datos para una evaluación robusta. "
        "Selecciona un período mayor o un activo con más historial."
    )


def render_methodology_note() -> None:
    """
    Nota metodológica breve.
    """
    st.info(
        "Metodología: los modelos se evalúan con división temporal para reducir fuga de información. "
        "Las señales se interpretan como scores probabilísticos, no como órdenes de compra o venta."
    )
