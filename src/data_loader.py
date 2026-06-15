import yfinance as yf
import pandas as pd
import numpy as np


def cargar_datos(ticker, periodo):
    df = yf.download(
        ticker,
        period=periodo,
        auto_adjust=True,
        progress=False
    )

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
