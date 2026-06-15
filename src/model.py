# src/model.py

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.8
):
    """
    Divide los datos respetando el orden temporal.
    No usa shuffle porque en series financieras no se debe mezclar pasado y futuro.
    """
    split_index = int(len(X) * train_size)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBRegressor:
    """
    Entrena un modelo XGBoost para estimar el precio del siguiente período.
    """
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_price_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evalúa el modelo contra un baseline naïve:
    precio futuro estimado = precio actual.
    """
    preds = model.predict(X_test)

    naive_preds = X_test["precio"].values

    mae_model = mean_absolute_error(y_test, preds)
    mae_naive = mean_absolute_error(y_test, naive_preds)

    rmse_model = np.sqrt(np.mean((y_test.values - preds) ** 2))
    mape_model = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100

    improvement_vs_naive = ((mae_naive - mae_model) / mae_naive) * 100

    real_direction = np.sign(y_test.values - X_test["precio"].values)
    predicted_direction = np.sign(preds - X_test["precio"].values)

    directional_accuracy = np.mean(real_direction == predicted_direction) * 100

    return {
        "predictions": preds,
        "naive_predictions": naive_preds,
        "mae_model": mae_model,
        "mae_naive": mae_naive,
        "rmse_model": rmse_model,
        "mape_model": mape_model,
        "improvement_vs_naive": improvement_vs_naive,
        "directional_accuracy": directional_accuracy,
    }


def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.8
) -> dict:
    """
    Pipeline completo:
    1. Divide temporalmente.
    2. Entrena XGBoost.
    3. Evalúa contra baseline naïve.
    4. Devuelve modelo, datos de prueba y métricas.
    """
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X,
        y,
        train_size=train_size
    )

    model = train_xgboost_regressor(X_train, y_train)

    metrics = evaluate_price_model(model, X_test, y_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
    }
