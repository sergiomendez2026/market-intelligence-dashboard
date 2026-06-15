# src/data_loader.py

import yfinance as yf
import pandas as pd


def load_market_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Descarga datos históricos de mercado desde yFinance.

    Parameters
    ----------
    ticker : str
        Símbolo financiero.
    period : str
        Ventana histórica. Ejemplo: '1y', '2y', '5y'.
    interval : str
        Frecuencia temporal. Ejemplo: '1d', '1h'.

    Returns
    -------
    pd.DataFrame
        Datos OHLCV limpios.
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No se encontraron datos para {ticker}")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.dropna()

    return df
