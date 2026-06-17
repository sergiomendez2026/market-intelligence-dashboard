# src/statistical_tests.py

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.metrics import accuracy_score, f1_score


def diebold_mariano_test(
    errors_model_a: np.ndarray,
    errors_model_b: np.ndarray,
    h: int = 1,
) -> dict:
    """
    Test Diebold-Mariano para comparar errores de forecasting.

    H0: ambos modelos tienen igual precisión predictiva.
    H1: existe diferencia significativa en precisión predictiva.

    Se compara la pérdida cuadrática:
    loss_a = error_a^2
    loss_b = error_b^2

    Si mean(loss_a - loss_b) > 0, el modelo B tiene menor pérdida promedio.
    """

    errors_model_a = np.asarray(errors_model_a, dtype=float)
    errors_model_b = np.asarray(errors_model_b, dtype=float)

    if len(errors_model_a) != len(errors_model_b):
        raise ValueError("Los errores de ambos modelos deben tener la misma longitud.")

    if len(errors_model_a) < 10:
        raise ValueError("Se requieren al menos 10 observaciones para Diebold-Mariano.")

    loss_a = errors_model_a ** 2
    loss_b = errors_model_b ** 2

    loss_diff = loss_a - loss_b
    mean_diff = np.mean(loss_diff)

    n = len(loss_diff)

    # Varianza tipo Newey-West simplificada para horizonte h
    gamma_0 = np.var(loss_diff, ddof=1)

    long_run_variance = gamma_0

    if h > 1:
        for lag in range(1, h):
            covariance = np.cov(loss_diff[lag:], loss_diff[:-lag], ddof=1)[0, 1]
            weight = 1 - lag / h
            long_run_variance += 2 * weight * covariance

    if long_run_variance <= 0:
        dm_statistic = 0.0
        p_value = 1.0
    else:
        dm_statistic = mean_diff / np.sqrt(long_run_variance / n)
        p_value = 2 * (1 - norm.cdf(abs(dm_statistic)))

    if p_value < 0.05:
        significance = "Significativo al 5%"
    else:
        significance = "No significativo al 5%"

    if mean_diff > 0:
        better_model = "Modelo B"
        interpretation = "El Modelo B presenta menor pérdida promedio."
    elif mean_diff < 0:
        better_model = "Modelo A"
        interpretation = "El Modelo A presenta menor pérdida promedio."
    else:
        better_model = "Empate"
        interpretation = "Ambos modelos presentan pérdida promedio equivalente."

    return {
        "Test": "Diebold-Mariano",
        "Métrica evaluada": "Error cuadrático",
        "Estadístico": round(float(dm_statistic), 6),
        "p-value": round(float(p_value), 6),
        "Media diferencia pérdida": round(float(mean_diff), 8),
        "Mejor modelo según pérdida": better_model,
        "Significancia": significance,
        "Interpretación": interpretation,
    }


def mcnemar_test(
    y_true: np.ndarray,
    predictions_model_a: np.ndarray,
    predictions_model_b: np.ndarray,
) -> dict:
    """
    Test de McNemar para comparar dos clasificadores sobre las mismas observaciones.

    H0: ambos modelos tienen la misma tasa de error.
    H1: existe diferencia significativa entre ambos modelos.
    """

    y_true = np.asarray(y_true).astype(int)
    predictions_model_a = np.asarray(predictions_model_a).astype(int)
    predictions_model_b = np.asarray(predictions_model_b).astype(int)

    if not (
        len(y_true) == len(predictions_model_a) == len(predictions_model_b)
    ):
        raise ValueError("y_true y las predicciones deben tener la misma longitud.")

    correct_a = predictions_model_a == y_true
    correct_b = predictions_model_b == y_true

    # b: A falla, B acierta
    b = np.sum((correct_a == False) & (correct_b == True))

    # c: A acierta, B falla
    c = np.sum((correct_a == True) & (correct_b == False))

    if b + c == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)

    if p_value < 0.05:
        significance = "Significativo al 5%"
    else:
        significance = "No significativo al 5%"

    if b > c:
        better_model = "Modelo B"
        interpretation = "El Modelo B acierta más casos donde el Modelo A falla."
    elif c > b:
        better_model = "Modelo A"
        interpretation = "El Modelo A acierta más casos donde el Modelo B falla."
    else:
        better_model = "Empate"
        interpretation = "No hay ventaja clara entre modelos."

    return {
        "Test": "McNemar",
        "b A falla/B acierta": int(b),
        "c A acierta/B falla": int(c),
        "Estadístico": round(float(statistic), 6),
        "p-value": round(float(p_value), 6),
        "Mejor modelo": better_model,
        "Significancia": significance,
        "Interpretación": interpretation,
    }


def bootstrap_metric_difference(
    y_true: np.ndarray,
    predictions_model_a: np.ndarray,
    predictions_model_b: np.ndarray,
    metric: str = "f1",
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap para estimar intervalo de confianza de la diferencia de métricas.

    Diferencia calculada:
    métrica(Modelo B) - métrica(Modelo A)
    """

    y_true = np.asarray(y_true).astype(int)
    predictions_model_a = np.asarray(predictions_model_a).astype(int)
    predictions_model_b = np.asarray(predictions_model_b).astype(int)

    if not (
        len(y_true) == len(predictions_model_a) == len(predictions_model_b)
    ):
        raise ValueError("y_true y las predicciones deben tener la misma longitud.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    differences = []

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, n)

        y_sample = y_true[sample_idx]
        pred_a_sample = predictions_model_a[sample_idx]
        pred_b_sample = predictions_model_b[sample_idx]

        if metric == "f1":
            score_a = f1_score(y_sample, pred_a_sample, zero_division=0)
            score_b = f1_score(y_sample, pred_b_sample, zero_division=0)
        elif metric == "accuracy":
            score_a = accuracy_score(y_sample, pred_a_sample)
            score_b = accuracy_score(y_sample, pred_b_sample)
        else:
            raise ValueError("metric debe ser 'f1' o 'accuracy'.")

        differences.append(score_b - score_a)

    differences = np.asarray(differences)

    mean_difference = np.mean(differences)
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)

    if ci_lower > 0 and ci_upper > 0:
        significance = "Modelo B superior con IC 95% positivo"
    elif ci_lower < 0 and ci_upper < 0:
        significance = "Modelo A superior con IC 95% negativo"
    else:
        significance = "Diferencia no concluyente con IC 95%"

    return {
        "Test": f"Bootstrap diferencia {metric.upper()}",
        "Diferencia media B-A": round(float(mean_difference * 100), 4),
        "IC 95% inferior": round(float(ci_lower * 100), 4),
        "IC 95% superior": round(float(ci_upper * 100), 4),
        "Significancia": significance,
    }


def compare_model_predictions_statistically(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ejecuta pruebas estadísticas entre Naive y Modelo Técnico.

    Columnas requeridas:
    - actual_return
    - actual_direction
    - naive_pred_return
    - naive_pred_direction
    - technical_pred_return
    - technical_pred_direction
    """

    required_columns = [
        "actual_return",
        "actual_direction",
        "naive_pred_return",
        "naive_pred_direction",
        "technical_pred_return",
        "technical_pred_direction",
    ]

    missing = [col for col in required_columns if col not in prediction_df.columns]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    actual_return = prediction_df["actual_return"].values
    actual_direction = prediction_df["actual_direction"].values

    naive_return = prediction_df["naive_pred_return"].values
    technical_return = prediction_df["technical_pred_return"].values

    naive_direction = prediction_df["naive_pred_direction"].values
    technical_direction = prediction_df["technical_pred_direction"].values

    errors_naive = actual_return - naive_return
    errors_technical = actual_return - technical_return

    dm_result = diebold_mariano_test(
        errors_model_a=errors_naive,
        errors_model_b=errors_technical,
    )

    mcnemar_result = mcnemar_test(
        y_true=actual_direction,
        predictions_model_a=naive_direction,
        predictions_model_b=technical_direction,
    )

    bootstrap_f1 = bootstrap_metric_difference(
        y_true=actual_direction,
        predictions_model_a=naive_direction,
        predictions_model_b=technical_direction,
        metric="f1",
    )

    bootstrap_accuracy = bootstrap_metric_difference(
        y_true=actual_direction,
        predictions_model_a=naive_direction,
        predictions_model_b=technical_direction,
        metric="accuracy",
    )

    results = [
        {
            "Comparación": "Naive vs Modelo Técnico",
            **dm_result,
        },
        {
            "Comparación": "Naive vs Modelo Técnico",
            **mcnemar_result,
        },
        {
            "Comparación": "Naive vs Modelo Técnico",
            **bootstrap_f1,
        },
        {
            "Comparación": "Naive vs Modelo Técnico",
            **bootstrap_accuracy,
        },
    ]

    return pd.DataFrame(results)
  
