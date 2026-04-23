"""Fit a model on the Titanic training set and publish it to MLflow.

Produces three things on every run:

1. A single ``pyfunc`` artifact (Preprocessor + estimator in one unit) logged
   to the current MLflow run and registered under the model name. A new
   version gets the ``candidate`` alias; ``--promote`` also moves the
   ``production`` alias.
2. ``models/preprocessor.pkl`` and ``models/<model>_model.pkl`` on disk for
   the Dockerfile and the filesystem-fallback serving path.
3. ``models/reference_stats.json`` — the training-set distribution
   snapshot the drift monitor compares live traffic against.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

# Cap joblib fan-out by default. ``n_jobs=-1`` spawns one worker per core,
# and each worker inherits the parent's resident dataset + imports; on a
# modern laptop with a dozen cores that can churn enough memory to tip an
# already-stressed kernel into OOM territory. Override via TITANIC_N_JOBS.
N_JOBS = int(os.getenv("TITANIC_N_JOBS", "2"))

from data_loader import load_data
from mlflow_model import log_and_register
from monitoring import ReferenceStats
from preprocessing import Preprocessor

MODELS_DIR = Path("models")
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"
REGISTERED_MODEL_NAME = "titanic-survival"
CANDIDATE_ALIAS = "candidate"
PRODUCTION_ALIAS = "production"

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


def _set_alias(client: MlflowClient, name: str, alias: str, version: str) -> None:
    """Set a registry alias, tolerating older MLflow clients."""
    client.set_registered_model_alias(name=name, alias=alias, version=version)


def train_model(model_name: str = "logistic_regression", *, promote: bool = False) -> float:
    MODELS_DIR.mkdir(exist_ok=True)

    train_df, _ = load_data()
    y = train_df["Survived"]

    preprocessor = Preprocessor().fit(train_df)
    X = preprocessor.transform(train_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        model = build_model(model_name)

        if model_name == "random_forest":
            grid = GridSearchCV(model, RF_PARAM_GRID, cv=5, scoring="accuracy", n_jobs=N_JOBS)
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

        # Filesystem artifacts (used by the Dockerfile / filesystem serving fallback
        # and by the pyfunc wrapper below as its constituent files).
        preprocessor.save(PREPROCESSOR_PATH)
        estimator_path = MODELS_DIR / f"{model_name}_model.pkl"
        joblib.dump(model, estimator_path)

        # Reference-distribution snapshot used by the live drift monitor.
        ReferenceStats.fit(train_df).save(REFERENCE_STATS_PATH)
        mlflow.log_artifact(str(REFERENCE_STATS_PATH))

        # Register a single pyfunc (Preprocessor + estimator in one atomic unit)
        # so serving loads one artifact by URI instead of two pickles by path.
        info = log_and_register(
            preprocessor_path=PREPROCESSOR_PATH,
            estimator_path=estimator_path,
            registered_model_name=REGISTERED_MODEL_NAME,
            code_paths=["src/preprocessing.py"],
        )

        client = MlflowClient()
        version = info.registered_model_version
        _set_alias(client, REGISTERED_MODEL_NAME, CANDIDATE_ALIAS, version)
        print(
            f"Registered {REGISTERED_MODEL_NAME} v{version} "
            f"(run {run.info.run_id}) with alias @{CANDIDATE_ALIAS}."
        )
        if promote:
            _set_alias(client, REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version)
            print(f"Promoted {REGISTERED_MODEL_NAME} v{version} to @{PRODUCTION_ALIAS}.")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Titanic model.")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Also set the `production` alias on this new version.",
    )
    args = parser.parse_args()
    train_model(model_name=args.model, promote=args.promote)
