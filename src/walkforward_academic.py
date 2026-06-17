# src/walkforward_academic.py

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

try:
    from statsmodels.tsa.arima.model import ARIMA

    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


def _safe_division(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcula RSI simple para construir variables técnicas básicas.
    """

    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def build_basic_technical_features(prices: pd.Series) -> pd.DataFrame:
    """
    Construye variables técnicas básicas desde una serie de precios.

    Estas variables sirven como base académica cuando no se pasan features
    más avanzadas desde el pipeline principal.
    """

    close = prices.dropna().astype(float)

    features = pd.DataFrame(index=close.index)
    features["return_1"] = close.pct_change()
    features["return_3"] = close.pct_change(3)
    features["return_5"] = close.pct_change(5)
    features["ma_20"] = close.rolling(20).mean()
    features["ma_50"] = close.rolling(50).mean()
    features["price_to_ma20"] = close / features["ma_20"] - 1
    features["price_to_ma50"] = close / features["ma_50"] - 1
    features["volatility_20"] = features["return_1"].rolling(20).std()
    features["momentum_10"] = close / close.shift(10) - 1
    features["rsi_14"] = calculate_rsi(close, window=14)

    return features.replace([np.inf, -np.inf], np.nan)


def build_supervised_financial_dataset(
    prices: pd.Series,
    features: pd.DataFrame | None = None,
    sentiment_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye dataset supervisado para predicción de retorno/dirección t+1.

    Cada fila usa información disponible en t para predecir retorno de t+1.
    """

    close = prices.dropna().astype(float)

    if features is None:
        X = build_basic_technical_features(close)
    else:
        X = features.copy()

    X = X.reindex(close.index)

    if sentiment_features is not None:
        sentiment = sentiment_features.reindex(close.index)
        X = pd.concat([X, sentiment], axis=1)

    dataset = X.copy()
    dataset["current_return"] = close.pct_change()
    dataset["future_return"] = close.pct_change().shift(-1)
    dataset["future_direction"] = (dataset["future_return"] > 0).astype(int)
    dataset["close"] = close

    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

    return dataset


def calculate_walkforward_metrics(
    y_true_return: np.ndarray,
    y_pred_return: np.ndarray,
    y_true_direction: np.ndarray,
    y_pred_direction: np.ndarray,
) -> dict:
    """
    Calcula métricas agregadas para validación walk-forward.
    """

    y_true_return = np.asarray(y_true_return)
    y_pred_return = np.asarray(y_pred_return)
    y_true_direction = np.asarray(y_true_direction)
    y_pred_direction = np.asarray(y_pred_direction)

    mae = mean_absolute_error(y_true_return, y_pred_return)
    rmse = np.sqrt(mean_squared_error(y_true_return, y_pred_return))

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_values = np.abs((y_true_return - y_pred_return) / y_true_return)
        mape_values = mape_values[np.isfinite(mape_values)]
        mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else np.nan

    metrics = {
        "MAE": round(float(mae), 6),
        "RMSE": round(float(rmse), 6),
        "MAPE": round(float(mape), 4) if not np.isnan(mape) else np.nan,
        "Accuracy": round(float(accuracy_score(y_true_direction, y_pred_direction) * 100), 2),
        "Precision": round(
            float(precision_score(y_true_direction, y_pred_direction, zero_division=0) * 100),
            2,
        ),
        "Recall": round(
            float(recall_score(y_true_direction, y_pred_direction, zero_division=0) * 100),
            2,
        ),
        "F1 Score": round(
            float(f1_score(y_true_direction, y_pred_direction, zero_division=0) * 100),
            2,
        ),
    }

    return metrics


def run_walkforward_naive(dataset: pd.DataFrame) -> dict:
    """
    Baseline Naive.

    Forecasting:
    precio(t+1) = precio(t), por tanto retorno esperado = 0.

    Dirección:
    usa la dirección del retorno actual como aproximación ingenua.
    """

    y_true_return = dataset["future_return"].values
    y_pred_return = np.zeros(len(dataset))

    y_true_direction = dataset["future_direction"].values
    y_pred_direction = (dataset["current_return"].values > 0).astype(int)

    metrics = calculate_walkforward_metrics(
        y_true_return=y_true_return,
        y_pred_return=y_pred_return,
        y_true_direction=y_true_direction,
        y_pred_direction=y_pred_direction,
    )

    metrics["Modelo"] = "Naive t+1 = precio actual"
    metrics["Tipo"] = "Baseline"
    metrics["Observaciones"] = len(dataset)

    return metrics


def run_walkforward_linear_regression(
    dataset: pd.DataFrame,
    initial_train_size: int = 252,
    test_window: int = 20,
    step_size: int = 20,
    max_windows: int = 12,
) -> dict:
    """
    Regresión lineal walk-forward.

    Entrena con variables técnicas y predice retorno futuro.
    La dirección predicha se obtiene según el signo del retorno predicho.
    """

    feature_columns = [
        col
        for col in dataset.columns
        if col not in ["future_return", "future_direction", "close"]
    ]

    y_true_return_all = []
    y_pred_return_all = []
    y_true_direction_all = []
    y_pred_direction_all = []

    n = len(dataset)
    windows_used = 0

    for train_end in range(initial_train_size, n - test_window, step_size):
        if windows_used >= max_windows:
            break

        train = dataset.iloc[:train_end]
        test = dataset.iloc[train_end : train_end + test_window]

        X_train = train[feature_columns]
        y_train = train["future_return"]

        X_test = test[feature_columns]
        y_test_return = test["future_return"].values
        y_test_direction = test["future_direction"].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_return = model.predict(X_test)
        y_pred_direction = (y_pred_return > 0).astype(int)

        y_true_return_all.extend(y_test_return)
        y_pred_return_all.extend(y_pred_return)
        y_true_direction_all.extend(y_test_direction)
        y_pred_direction_all.extend(y_pred_direction)

        windows_used += 1

    metrics = calculate_walkforward_metrics(
        y_true_return=np.asarray(y_true_return_all),
        y_pred_return=np.asarray(y_pred_return_all),
        y_true_direction=np.asarray(y_true_direction_all),
        y_pred_direction=np.asarray(y_pred_direction_all),
    )

    metrics["Modelo"] = "Regresión lineal"
    metrics["Tipo"] = "Baseline interpretable"
    metrics["Ventanas"] = windows_used
    metrics["Observaciones"] = len(y_true_return_all)

    return metrics


def run_walkforward_arima(
    prices: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    initial_train_size: int = 252,
    test_window: int = 20,
    step_size: int = 20,
    max_windows: int = 8,
) -> dict:
    """
    ARIMA walk-forward para forecasting de precios.

    La dirección se calcula comparando el precio forecast contra el precio actual.
    """

    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels no está instalado. Agrega statsmodels a requirements.txt.")

    close = prices.dropna().astype(float)

    y_true_return_all = []
    y_pred_return_all = []
    y_true_direction_all = []
    y_pred_direction_all = []

    n = len(close)
    windows_used = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for train_end in range(initial_train_size, n - test_window - 1, step_size):
            if windows_used >= max_windows:
                break

            history = list(close.iloc[:train_end].values)

            for i in range(test_window):
                current_position = train_end + i

                if current_position + 1 >= n:
                    break

                current_price = close.iloc[current_position]
                next_price = close.iloc[current_position + 1]

                try:
                    model = ARIMA(history, order=order)
                    fitted = model.fit()
                    forecast_price = fitted.forecast(steps=1)[0]
                except Exception:
                    forecast_price = history[-1]

                true_return = (next_price / current_price) - 1
                pred_return = (forecast_price / current_price) - 1

                y_true_return_all.append(true_return)
                y_pred_return_all.append(pred_return)
                y_true_direction_all.append(int(true_return > 0))
                y_pred_direction_all.append(int(pred_return > 0))

                history.append(next_price)

            windows_used += 1

    metrics = calculate_walkforward_metrics(
        y_true_return=np.asarray(y_true_return_all),
        y_pred_return=np.asarray(y_pred_return_all),
        y_true_direction=np.asarray(y_true_direction_all),
        y_pred_direction=np.asarray(y_pred_direction_all),
    )

    metrics["Modelo"] = f"ARIMA{order}"
    metrics["Tipo"] = "Baseline econométrico"
    metrics["Ventanas"] = windows_used
    metrics["Observaciones"] = len(y_true_return_all)

    return metrics


def run_walkforward_technical_model(
    dataset: pd.DataFrame,
    initial_train_size: int = 252,
    test_window: int = 20,
    step_size: int = 20,
    max_windows: int = 12,
    model_name: str = "Modelo técnico sin sentimiento",
) -> dict:
    """
    Modelo técnico walk-forward.

    Usa Random Forest como clasificador robusto inicial.
    Más adelante puede reemplazarse o complementarse con XGBoost.
    """

    feature_columns = [
        col
        for col in dataset.columns
        if col not in ["future_return", "future_direction", "close"]
    ]

    y_true_return_all = []
    y_pred_return_all = []
    y_true_direction_all = []
    y_pred_direction_all = []

    n = len(dataset)
    windows_used = 0

    for train_end in range(initial_train_size, n - test_window, step_size):
        if windows_used >= max_windows:
            break

        train = dataset.iloc[:train_end]
        test = dataset.iloc[train_end : train_end + test_window]

        X_train = train[feature_columns]
        y_train = train["future_direction"]

        X_test = test[feature_columns]
        y_test_return = test["future_return"].values
        y_test_direction = test["future_direction"].values

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        y_pred_direction = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            y_pred_return = proba - 0.5
        else:
            y_pred_return = y_pred_direction - 0.5

        y_true_return_all.extend(y_test_return)
        y_pred_return_all.extend(y_pred_return)
        y_true_direction_all.extend(y_test_direction)
        y_pred_direction_all.extend(y_pred_direction)

        windows_used += 1

    metrics = calculate_walkforward_metrics(
        y_true_return=np.asarray(y_true_return_all),
        y_pred_return=np.asarray(y_pred_return_all),
        y_true_direction=np.asarray(y_true_direction_all),
        y_pred_direction=np.asarray(y_pred_direction_all),
    )

    metrics["Modelo"] = model_name
    metrics["Tipo"] = "Machine Learning"
    metrics["Ventanas"] = windows_used
    metrics["Observaciones"] = len(y_true_return_all)

    return metrics


def compare_walkforward_academic_models(
    prices: pd.Series,
    features: pd.DataFrame | None = None,
    sentiment_features: pd.DataFrame | None = None,
    initial_train_size: int = 252,
    test_window: int = 20,
    step_size: int = 20,
    max_windows: int = 12,
    include_arima: bool = True,
) -> pd.DataFrame:
    """
    Compara modelos académicos bajo walk-forward validation.

    Incluye:
    - Naive.
    - Regresión lineal.
    - ARIMA.
    - Modelo técnico sin sentimiento.
    - Modelo técnico + sentimiento, si se entregan sentiment_features.
    """

    technical_dataset = build_supervised_financial_dataset(
        prices=prices,
        features=features,
        sentiment_features=None,
    )

    results = []

    try:
        results.append(run_walkforward_naive(technical_dataset))
    except Exception as error:
        results.append({"Modelo": "Naive t+1 = precio actual", "Error": str(error)})

    try:
        results.append(
            run_walkforward_linear_regression(
                dataset=technical_dataset,
                initial_train_size=initial_train_size,
                test_window=test_window,
                step_size=step_size,
                max_windows=max_windows,
            )
        )
    except Exception as error:
        results.append({"Modelo": "Regresión lineal", "Error": str(error)})

    if include_arima:
        try:
            results.append(
                run_walkforward_arima(
                    prices=prices,
                    initial_train_size=initial_train_size,
                    test_window=test_window,
                    step_size=step_size,
                    max_windows=min(max_windows, 8),
                )
            )
        except Exception as error:
            results.append({"Modelo": "ARIMA(1,1,1)", "Error": str(error)})

    try:
        results.append(
            run_walkforward_technical_model(
                dataset=technical_dataset,
                initial_train_size=initial_train_size,
                test_window=test_window,
                step_size=step_size,
                max_windows=max_windows,
                model_name="Modelo técnico sin sentimiento",
            )
        )
    except Exception as error:
        results.append({"Modelo": "Modelo técnico sin sentimiento", "Error": str(error)})

    if sentiment_features is not None:
        try:
            sentiment_dataset = build_supervised_financial_dataset(
                prices=prices,
                features=features,
                sentiment_features=sentiment_features,
            )

            results.append(
                run_walkforward_technical_model(
                    dataset=sentiment_dataset,
                    initial_train_size=initial_train_size,
                    test_window=test_window,
                    step_size=step_size,
                    max_windows=max_windows,
                    model_name="Modelo técnico + sentimiento",
                )
            )
        except Exception as error:
            results.append({"Modelo": "Modelo técnico + sentimiento", "Error": str(error)})

    results_df = pd.DataFrame(results)

    preferred_columns = [
        "Modelo",
        "Tipo",
        "F1 Score",
        "Accuracy",
        "Precision",
        "Recall",
        "MAE",
        "RMSE",
        "MAPE",
        "Ventanas",
        "Observaciones",
        "Error",
    ]

    available_columns = [col for col in preferred_columns if col in results_df.columns]
    remaining_columns = [col for col in results_df.columns if col not in available_columns]

    return results_df[available_columns + remaining_columns]

def collect_walkforward_predictions_for_statistics(
    prices: pd.Series,
    features: pd.DataFrame | None = None,
    initial_train_size: int = 252,
    test_window: int = 20,
    step_size: int = 20,
    max_windows: int = 8,
) -> pd.DataFrame:
    """
    Genera predicciones walk-forward para pruebas estadísticas.

    Compara:
    - Naive
    - Modelo técnico sin sentimiento

    Más adelante se podrá extender a:
    - Modelo técnico + FinBERT
    """

    dataset = build_supervised_financial_dataset(
        prices=prices,
        features=features,
        sentiment_features=None,
    )

    feature_columns = [
        col
        for col in dataset.columns
        if col not in ["future_return", "future_direction", "close"]
    ]

    records = []
    n = len(dataset)
    windows_used = 0

    for train_end in range(initial_train_size, n - test_window, step_size):
        if windows_used >= max_windows:
            break

        train = dataset.iloc[:train_end]
        test = dataset.iloc[train_end : train_end + test_window]

        X_train = train[feature_columns]
        y_train = train["future_direction"]

        X_test = test[feature_columns]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        technical_pred_direction = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            technical_pred_return = model.predict_proba(X_test)[:, 1] - 0.5
        else:
            technical_pred_return = technical_pred_direction - 0.5

        naive_pred_direction = (test["current_return"].values > 0).astype(int)
        naive_pred_return = np.zeros(len(test))

        for i in range(len(test)):
            records.append(
                {
                    "date": test.index[i],
                    "actual_return": float(test["future_return"].iloc[i]),
                    "actual_direction": int(test["future_direction"].iloc[i]),
                    "naive_pred_return": float(naive_pred_return[i]),
                    "naive_pred_direction": int(naive_pred_direction[i]),
                    "technical_pred_return": float(technical_pred_return[i]),
                    "technical_pred_direction": int(technical_pred_direction[i]),
                    "window": windows_used + 1,
                }
            )

        windows_used += 1

    return pd.DataFrame(records)
