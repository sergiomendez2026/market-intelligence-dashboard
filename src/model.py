# src/model.py

import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


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


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBClassifier:
    """
    Entrena un modelo XGBoost para predecir la dirección del siguiente período.
    1 = sube
    0 = baja o no sube
    """
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
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
    Evalúa el modelo de regresión contra un baseline naïve:
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


def evaluate_direction_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evalúa el modelo de clasificación direccional.
    """
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds) * 100
    precision = precision_score(y_test, preds, zero_division=0) * 100
    recall = recall_score(y_test, preds, zero_division=0) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100

    baseline_direction = int(y_test.iloc[:len(y_test)].mode()[0])
    baseline_preds = np.full(shape=len(y_test), fill_value=baseline_direction)

    baseline_accuracy = accuracy_score(y_test, baseline_preds) * 100

    improvement_vs_direction_baseline = accuracy - baseline_accuracy

    return {
        "direction_predictions": preds,
        "direction_probabilities": proba,
        "direction_accuracy": accuracy,
        "direction_precision": precision,
        "direction_recall": recall,
        "direction_f1": f1,
        "direction_baseline_accuracy": baseline_accuracy,
        "improvement_vs_direction_baseline": improvement_vs_direction_baseline,
    }


def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.8
) -> dict:
    """
    Pipeline de regresión:
    1. Divide temporalmente.
    2. Entrena XGBoost Regressor.
    3. Evalúa contra baseline naïve.
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


def train_and_evaluate_direction_model(
    X: pd.DataFrame,
    y_direction: pd.Series,
    train_size: float = 0.8
) -> dict:
    """
    Pipeline de clasificación direccional:
    1. Divide temporalmente.
    2. Entrena XGBoost Classifier.
    3. Evalúa accuracy, precision, recall y F1.
    """
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X,
        y_direction,
        train_size=train_size
    )

    model = train_xgboost_classifier(X_train, y_train)

    metrics = evaluate_direction_model(model, X_test, y_test)

    return {
        "direction_model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "direction_metrics": metrics,
    }
