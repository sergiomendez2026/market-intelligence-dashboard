# src/validation.py

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


def walk_forward_regression_validation(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 120,
    test_size: int = 20,
    step_size: int = 20
) -> dict:
    """
    Validación walk-forward para modelo de regresión financiera.

    Entrena con una ventana histórica inicial y evalúa en bloques futuros.
    Reduce el riesgo de sobreestimar desempeño por una única partición temporal.
    """

    predictions = []
    actuals = []
    naive_predictions = []
    test_indices = []

    n_samples = len(X)

    if n_samples < min_train_size + test_size:
        return {
            "available": False,
            "message": "No hay suficientes datos para walk-forward validation."
        }

    for train_end in range(min_train_size, n_samples - test_size + 1, step_size):
        test_start = train_end
        test_end = train_end + test_size

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

        model.fit(X_train, y_train)

        fold_preds = model.predict(X_test)
        fold_naive = X_test["precio"].values

        predictions.extend(fold_preds)
        actuals.extend(y_test.values)
        naive_predictions.extend(fold_naive)
        test_indices.extend(y_test.index)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    naive_predictions = np.array(naive_predictions)

    mae_model = mean_absolute_error(actuals, predictions)
    mae_naive = mean_absolute_error(actuals, naive_predictions)

    rmse_model = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape_model = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    improvement_vs_naive = ((mae_naive - mae_model) / mae_naive) * 100

    real_direction = np.sign(actuals - naive_predictions)
    predicted_direction = np.sign(predictions - naive_predictions)

    directional_accuracy = np.mean(real_direction == predicted_direction) * 100

    results_df = pd.DataFrame(
        {
            "actual": actuals,
            "prediction": predictions,
            "naive_prediction": naive_predictions,
            "error": actuals - predictions,
            "absolute_error": np.abs(actuals - predictions),
        },
        index=test_indices
    )

    return {
        "available": True,
        "results": results_df,
        "metrics": {
            "mae_model": mae_model,
            "mae_naive": mae_naive,
            "rmse_model": rmse_model,
            "mape_model": mape_model,
            "improvement_vs_naive": improvement_vs_naive,
            "directional_accuracy": directional_accuracy,
            "n_predictions": len(predictions),
        }
    }


def walk_forward_direction_validation(
    X: pd.DataFrame,
    y_direction: pd.Series,
    min_train_size: int = 120,
    test_size: int = 20,
    step_size: int = 20
) -> dict:
    """
    Validación walk-forward para modelo direccional.

    Evalúa si el modelo predice correctamente la dirección futura:
    1 = sube
    0 = baja o no sube.
    """

    predictions = []
    probabilities = []
    actuals = []
    test_indices = []

    n_samples = len(X)

    if n_samples < min_train_size + test_size:
        return {
            "available": False,
            "message": "No hay suficientes datos para walk-forward direccional."
        }

    for train_end in range(min_train_size, n_samples - test_size + 1, step_size):
        test_start = train_end
        test_end = train_end + test_size

        X_train = X.iloc[:train_end]
        y_train = y_direction.iloc[:train_end]

        X_test = X.iloc[test_start:test_end]
        y_test = y_direction.iloc[test_start:test_end]

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        fold_preds = model.predict(X_test)
        fold_proba = model.predict_proba(X_test)[:, 1]

        predictions.extend(fold_preds)
        probabilities.extend(fold_proba)
        actuals.extend(y_test.values)
        test_indices.extend(y_test.index)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    actuals = np.array(actuals)

    majority_class = int(pd.Series(actuals).mode()[0])
    baseline_preds = np.full(shape=len(actuals), fill_value=majority_class)

    accuracy = accuracy_score(actuals, predictions) * 100
    precision = precision_score(actuals, predictions, zero_division=0) * 100
    recall = recall_score(actuals, predictions, zero_division=0) * 100
    f1 = f1_score(actuals, predictions, zero_division=0) * 100

    baseline_accuracy = accuracy_score(actuals, baseline_preds) * 100
    improvement_vs_baseline = accuracy - baseline_accuracy

    results_df = pd.DataFrame(
        {
            "actual_direction": actuals,
            "predicted_direction": predictions,
            "up_probability": probabilities,
        },
        index=test_indices
    )

    return {
        "available": True,
        "results": results_df,
        "metrics": {
            "direction_accuracy": accuracy,
            "direction_precision": precision,
            "direction_recall": recall,
            "direction_f1": f1,
            "direction_baseline_accuracy": baseline_accuracy,
            "improvement_vs_direction_baseline": improvement_vs_baseline,
            "n_predictions": len(predictions),
        }
    }
