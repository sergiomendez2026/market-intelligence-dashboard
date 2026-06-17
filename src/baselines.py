# src/baselines.py

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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


@dataclass
class BaselineResult:
    """
    Contenedor estándar para resultados de modelos baseline.
    """

    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_true_direction: np.ndarray
    y_pred_direction: np.ndarray
    metrics: dict


def calculate_direction(values: np.ndarray) -> np.ndarray:
    """
    Convierte una serie de valores en dirección binaria.

    1 = sube o se mantiene positivo
    0 = baja o no es positivo
    """

    values = np.asarray(values)

    return (values > 0).astype(int)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calcula métricas de forecasting/regresión.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_values = np.abs((y_true - y_pred) / y_true)
        mape_values = mape_values[np.isfinite(mape_values)]

        if len(mape_values) == 0:
            mape = np.nan
        else:
            mape = np.mean(mape_values) * 100

    return {
        "MAE": round(float(mae), 6),
        "RMSE": round(float(rmse), 6),
        "MAPE": round(float(mape), 4) if not np.isnan(mape) else np.nan,
    }


def calculate_directional_metrics(
    y_true_direction: np.ndarray,
    y_pred_direction: np.ndarray,
) -> dict:
    """
    Calcula métricas de clasificación direccional.
    """

    y_true_direction = np.asarray(y_true_direction)
    y_pred_direction = np.asarray(y_pred_direction)

    return {
        "Accuracy": round(
            float(accuracy_score(y_true_direction, y_pred_direction) * 100),
            2,
        ),
        "Precision": round(
            float(
                precision_score(
                    y_true_direction,
                    y_pred_direction,
                    zero_division=0,
                )
                * 100
            ),
            2,
        ),
        "Recall": round(
            float(
                recall_score(
                    y_true_direction,
                    y_pred_direction,
                    zero_division=0,
                )
                * 100
            ),
            2,
        ),
        "F1 Score": round(
            float(
                f1_score(
                    y_true_direction,
                    y_pred_direction,
                    zero_division=0,
                )
                * 100
            ),
            2,
        ),
    }


def run_naive_price_baseline(
    prices: pd.Series,
) -> BaselineResult:
    """
    Baseline naive para forecasting:

    predicción(t+1) = precio(t)

    Este modelo asume que el precio futuro será igual al precio actual.
    Es un baseline obligatorio en predicción financiera.
    """

    clean_prices = prices.dropna().astype(float)

    if len(clean_prices) < 3:
        raise ValueError("Se requieren al menos 3 observaciones para el baseline naive.")

    y_true = clean_prices.iloc[1:].values
    y_pred = clean_prices.iloc[:-1].values

    true_returns = pd.Series(y_true).pct_change().fillna(0).values
    pred_returns = pd.Series(y_pred).pct_change().fillna(0).values

    y_true_direction = calculate_direction(true_returns)
    y_pred_direction = calculate_direction(pred_returns)

    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    directional_metrics = calculate_directional_metrics(
        y_true_direction,
        y_pred_direction,
    )

    metrics = {
        **regression_metrics,
        **directional_metrics,
        "Model": "Naive t+1 = precio actual",
    }

    return BaselineResult(
        model_name="Naive t+1 = precio actual",
        y_true=y_true,
        y_pred=y_pred,
        y_true_direction=y_true_direction,
        y_pred_direction=y_pred_direction,
        metrics=metrics,
    )


def run_linear_regression_baseline(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
) -> BaselineResult:
    """
    Baseline de regresión lineal.

    Usa las mismas variables técnicas que los modelos más complejos,
    pero con una relación lineal simple.
    """

    data = pd.concat([features, target.rename("target")], axis=1).dropna()

    if len(data) < 30:
        raise ValueError("Se requieren al menos 30 observaciones para regresión lineal.")

    X = data.drop(columns=["target"])
    y = data["target"]

    split_index = int(len(data) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_true_direction = calculate_direction(y_test.values)
    y_pred_direction = calculate_direction(y_pred)

    regression_metrics = calculate_regression_metrics(y_test.values, y_pred)
    directional_metrics = calculate_directional_metrics(
        y_true_direction,
        y_pred_direction,
    )

    metrics = {
        **regression_metrics,
        **directional_metrics,
        "Model": "Regresión lineal",
    }

    return BaselineResult(
        model_name="Regresión lineal",
        y_true=y_test.values,
        y_pred=y_pred,
        y_true_direction=y_true_direction,
        y_pred_direction=y_pred_direction,
        metrics=metrics,
    )


def run_arima_baseline(
    prices: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    test_size: float = 0.2,
) -> BaselineResult:
    """
    Baseline ARIMA para forecasting de precios.

    ARIMA se usa como baseline econométrico clásico.
    """

    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels no está instalado. Agrega 'statsmodels' a requirements.txt."
        )

    clean_prices = prices.dropna().astype(float)

    if len(clean_prices) < 50:
        raise ValueError("Se requieren al menos 50 observaciones para ARIMA.")

    split_index = int(len(clean_prices) * (1 - test_size))

    train = clean_prices.iloc[:split_index]
    test = clean_prices.iloc[split_index:]

    history = list(train.values)
    predictions = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for actual_value in test.values:
            try:
                model = ARIMA(history, order=order)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=1)[0]
            except Exception:
                forecast = history[-1]

            predictions.append(forecast)
            history.append(actual_value)

    y_true = test.values
    y_pred = np.asarray(predictions)

    true_returns = pd.Series(y_true).pct_change().fillna(0).values
    pred_returns = pd.Series(y_pred).pct_change().fillna(0).values

    y_true_direction = calculate_direction(true_returns)
    y_pred_direction = calculate_direction(pred_returns)

    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    directional_metrics = calculate_directional_metrics(
        y_true_direction,
        y_pred_direction,
    )

    metrics = {
        **regression_metrics,
        **directional_metrics,
        "Model": f"ARIMA{order}",
    }

    return BaselineResult(
        model_name=f"ARIMA{order}",
        y_true=y_true,
        y_pred=y_pred,
        y_true_direction=y_true_direction,
        y_pred_direction=y_pred_direction,
        metrics=metrics,
    )


def compare_academic_baselines(
    prices: pd.Series,
    features: pd.DataFrame | None = None,
    target: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Ejecuta y compara baselines académicos.

    Incluye:
    - Naive
    - Regresión lineal, si se entregan features y target
    - ARIMA
    """

    results = []

    try:
        naive_result = run_naive_price_baseline(prices)
        results.append(naive_result.metrics)
    except Exception as error:
        results.append(
            {
                "Model": "Naive t+1 = precio actual",
                "Error": str(error),
            }
        )

    if features is not None and target is not None:
        try:
            linear_result = run_linear_regression_baseline(features, target)
            results.append(linear_result.metrics)
        except Exception as error:
            results.append(
                {
                    "Model": "Regresión lineal",
                    "Error": str(error),
                }
            )

    try:
        arima_result = run_arima_baseline(prices)
        results.append(arima_result.metrics)
    except Exception as error:
        results.append(
            {
                "Model": "ARIMA(1,1,1)",
                "Error": str(error),
            }
        )

    return pd.DataFrame(results)
