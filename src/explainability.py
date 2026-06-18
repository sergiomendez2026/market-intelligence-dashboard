# src/explainability.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def calculate_random_forest_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula importancia de variables usando Random Forest.

    Esta función permite explicar qué variables influyen más en la predicción
    direccional del modelo.
    """

    if X.empty:
        raise ValueError("X está vacío. No se puede calcular importancia de variables.")

    if len(X) != len(y):
        raise ValueError("X e y deben tener la misma longitud.")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )

    model.fit(X, y)

    importance_df = pd.DataFrame(
        {
            "Variable": X.columns,
            "Importancia": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importancia",
        ascending=False,
    ).reset_index(drop=True)

    total_importance = importance_df["Importancia"].sum()

    if total_importance > 0:
        importance_df["Importancia (%)"] = (
            importance_df["Importancia"] / total_importance * 100
        )
    else:
        importance_df["Importancia (%)"] = 0

    importance_df["Importancia (%)"] = importance_df["Importancia (%)"].round(2)
    importance_df["Importancia"] = importance_df["Importancia"].round(6)

    return importance_df


def classify_feature_group(variable_name: str) -> str:
    """
    Clasifica una variable según su familia metodológica.
    """

    variable = variable_name.lower()

    if "finbert" in variable or "sentiment" in variable:
        return "Sentimiento FinBERT"

    if "rsi" in variable:
        return "Momentum / RSI"

    if "ma" in variable or "moving" in variable:
        return "Medias móviles"

    if "volatility" in variable or "std" in variable:
        return "Volatilidad"

    if "return" in variable or "momentum" in variable:
        return "Retornos / Momentum"

    if "price" in variable or "close" in variable:
        return "Precio"

    return "Otras variables"


def summarize_feature_importance_by_group(
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrupa la importancia de variables por familia metodológica.
    """

    if importance_df.empty:
        return pd.DataFrame()

    grouped_df = importance_df.copy()

    grouped_df["Grupo"] = grouped_df["Variable"].apply(classify_feature_group)

    summary = (
        grouped_df.groupby("Grupo", as_index=False)["Importancia (%)"]
        .sum()
        .sort_values(by="Importancia (%)", ascending=False)
        .reset_index(drop=True)
    )

    summary["Importancia (%)"] = summary["Importancia (%)"].round(2)

    return summary


def get_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Devuelve las principales variables del modelo.
    """

    if importance_df.empty:
        return pd.DataFrame()

    return importance_df.head(top_n).copy()
