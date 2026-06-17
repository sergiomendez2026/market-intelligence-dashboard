# src/model_comparison.py

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.8
):
    split_index = int(len(X) * train_size)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds) * 100
    precision = precision_score(y_test, preds, zero_division=0) * 100
    recall = recall_score(y_test, preds, zero_division=0) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def compare_direction_models(
    X: pd.DataFrame,
    y_direction: pd.Series,
    train_size: float = 0.8
) -> pd.DataFrame:
    """
    Compara modelos de clasificación direccional usando split temporal.

    Modelos:
    - Baseline Dummy
    - Logistic Regression
    - Random Forest
    - XGBoost Classifier
    """

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X,
        y_direction,
        train_size=train_size
    )

    models = {
        "Baseline Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test)

        results.append({
            "Modelo": model_name,
            "Accuracy": round(metrics["accuracy"], 2),
            "Precision": round(metrics["precision"], 2),
            "Recall": round(metrics["recall"], 2),
            "F1 Score": round(metrics["f1_score"], 2),
        })

    comparison_df = pd.DataFrame(results)

    comparison_df["Exceso Accuracy vs Baseline"] = (
        comparison_df["Accuracy"] - comparison_df.loc[
            comparison_df["Modelo"] == "Baseline Dummy",
            "Accuracy"
        ].iloc[0]
    ).round(2)

    return comparison_df
  
