"""Evaluate a persisted model on a stratified holdout from train.csv.

The previous version read a `data/validation.csv` that the pipeline never
produced. This version splits from train.csv with the same seed as training
so the holdout is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessor


def main(model_path: str, preprocessor_path: str, train_csv: str) -> None:
    model = joblib.load(model_path)
    preprocessor = Preprocessor.load(preprocessor_path)

    df = pd.read_csv(train_csv)
    y = df["Survived"]
    X = preprocessor.transform(df)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path("models") / "random_forest_model.pkl"))
    parser.add_argument("--preprocessor", default=str(Path("models") / "preprocessor.pkl"))
    parser.add_argument("--train_csv", default=str(Path("data") / "train.csv"))
    args = parser.parse_args()
    main(args.model, args.preprocessor, args.train_csv)
