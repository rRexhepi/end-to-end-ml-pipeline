"""Fit a model on the Titanic training set and log the run with MLflow.

The fitted `Preprocessor` is logged as an MLflow artifact and also written
to `models/preprocessor.pkl` so the serving API can load both from disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from data_loader import load_data
from preprocessing import Preprocessor

MODELS_DIR = Path("models")
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}


def build_model(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(solver="liblinear", random_state=42)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=42)
    raise ValueError(f"Unknown model: {model_name!r}")


def train_model(model_name: str = "logistic_regression") -> float:
    MODELS_DIR.mkdir(exist_ok=True)

    train_df, _ = load_data()
    y = train_df["Survived"]

    preprocessor = Preprocessor().fit(train_df)
    X = preprocessor.transform(train_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = build_model(model_name)

        if model_name == "random_forest":
            grid = GridSearchCV(model, RF_PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        print(f"Model: {model_name}")
        print(f"Validation Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", classification_report(y_val, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, f"{model_name}-model")

        preprocessor.save(PREPROCESSOR_PATH)
        mlflow.log_artifact(str(PREPROCESSOR_PATH))
        joblib.dump(model, MODELS_DIR / f"{model_name}_model.pkl")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Titanic model.")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
    )
    args = parser.parse_args()
    train_model(model_name=args.model)
