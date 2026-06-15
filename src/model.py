# src/model.py

import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score


def time_series_split(df: pd.DataFrame, train_size: float = 0.8):
    split_idx = int(len(df) * train_size)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def train_price_model(train: pd.DataFrame, feature_cols: list[str]):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    X_train = train[feature_cols]
    y_train = train["target_price_next"]

    model.fit(X_train, y_train)
    return model


def evaluate_price_model(model, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    X_test = test[feature_cols]
    y_test = test["target_price_next"]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    naive_preds = test["Close"]
    naive_mae = mean_absolute_error(y_test, naive_preds)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Naive_MAE": naive_mae,
        "Improvement_vs_Naive_%": ((naive_mae - mae) / naive_mae) * 100,
    }


def train_direction_model(train: pd.DataFrame, feature_cols: list[str]):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    X_train = train[feature_cols]
    y_train = train["target_direction"]

    model.fit(X_train, y_train)
    return model


def evaluate_direction_model(model, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    X_test = test[feature_cols]
    y_test = test["target_direction"]

    preds = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "F1": f1_score(y_test, preds),
    }
