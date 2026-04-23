"""Run the full training + inference pipeline.

Fits a `Preprocessor` and RandomForest on train.csv, evaluates on a stratified
20% holdout, and scores test.csv into predictions/submission.csv.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from preprocessing import Preprocessor

MODELS_DIR = Path("models")
PREDICTIONS_DIR = Path("predictions")

# See train_model.py for why we don't default to n_jobs=-1.
N_JOBS = int(os.getenv("TITANIC_N_JOBS", "2"))

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv("data/train.csv")
    y = train_df["Survived"]

    preprocessor = Preprocessor().fit(train_df)
    X = preprocessor.transform(train_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Tuning RandomForest...")
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_PARAM_GRID,
        cv=5,
        n_jobs=N_JOBS,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_val)
    print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    preprocessor.save(MODELS_DIR / "preprocessor.pkl")
    joblib.dump(model, MODELS_DIR / "random_forest_model.pkl")

    test_df = pd.read_csv("data/test.csv")
    X_test = preprocessor.transform(test_df)
    predictions = model.predict(X_test)

    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": predictions}
    )
    submission.to_csv(PREDICTIONS_DIR / "submission.csv", index=False)
    print(f"\nWrote {len(submission)} predictions to {PREDICTIONS_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
