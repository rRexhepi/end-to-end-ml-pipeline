"""Batch scoring CLI: takes an input CSV, writes predictions CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import Preprocessor


def make_predictions(input_csv: str, output_csv: str, model_path: str, preprocessor_path: str) -> None:
    model = joblib.load(model_path)
    preprocessor = Preprocessor.load(preprocessor_path)

    data = pd.read_csv(input_csv)
    X = preprocessor.transform(data)
    data["Predictions"] = model.predict(X)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False)
    print(f"Wrote {len(data)} predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a CSV with the trained Titanic model.")
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_data", required=True)
    parser.add_argument("--model", default="models/random_forest_model.pkl")
    parser.add_argument("--preprocessor", default="models/preprocessor.pkl")
    args = parser.parse_args()
    make_predictions(args.input_data, args.output_data, args.model, args.preprocessor)
