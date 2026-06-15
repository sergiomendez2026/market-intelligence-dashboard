import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


@st.cache_data(ttl=3600)
def cargar_datos(ticker: str, periodo: str) -> pd.Series:
    df = yf.download(
        ticker,
        period=periodo,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No se encontraron datos para {ticker}")

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    arr = np.array(close).flatten().astype(float)

    serie = pd.Series(
        arr,
        index=df.index[-len(arr):],
        name="Close"
    )

    return serie.dropna()
