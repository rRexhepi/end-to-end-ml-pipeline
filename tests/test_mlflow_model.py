"""Round-trip test for the MLflow pyfunc wrapper.

Save the Preprocessor + estimator as a single pyfunc, load it back via
``mlflow.pyfunc.load_model``, and verify that predictions on raw passenger
rows match what the unwrapped estimator gives. This pins the property
that matters: serving and training share one artifact, so they can't drift
apart.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlflow_model import TitanicSurvivalModel
from preprocessing import Preprocessor


def _sample_passengers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Pclass": 3, "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
            {"Pclass": 1, "Sex": "female", "Age": 38.0, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C"},
            {"Pclass": 2, "Sex": "female", "Age": None, "SibSp": 0, "Parch": 0, "Fare": None, "Embarked": "S"},
        ]
    )


def test_pyfunc_round_trip_matches_raw_estimator(train_frame, tmp_path, monkeypatch):
    # Point MLflow at an isolated tracking store so the test doesn't leak into ./mlruns.
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    preprocessor = Preprocessor().fit(train_frame)
    X = preprocessor.transform(train_frame)
    estimator = LogisticRegression(max_iter=200).fit(X, train_frame["Survived"])

    preprocessor_path = tmp_path / "preprocessor.pkl"
    estimator_path = tmp_path / "estimator.pkl"
    preprocessor.save(preprocessor_path)
    joblib.dump(estimator, estimator_path)

    # Log the pyfunc into a dedicated MLflow run so we can round-trip via URI.
    with mlflow.start_run() as run:
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TitanicSurvivalModel(),
            artifacts={
                "preprocessor": str(preprocessor_path),
                "estimator": str(estimator_path),
            },
            code_paths=[str(Path("src") / "preprocessing.py")],
        )

    loaded = mlflow.pyfunc.load_model(info.model_uri)

    passengers = _sample_passengers()
    pyfunc_preds = loaded.predict(passengers).tolist()
    direct_preds = estimator.predict(preprocessor.transform(passengers)).tolist()

    assert pyfunc_preds == direct_preds
    assert run.info.run_id  # run completed cleanly
