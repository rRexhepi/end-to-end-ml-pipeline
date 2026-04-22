"""MLflow ``pyfunc`` wrapper around the fitted Preprocessor + estimator.

Serving used to load two pickles off disk: ``preprocessor.pkl`` and the
sklearn estimator. That's two versions to keep in sync and two artifacts
to get wrong. This module wraps both into a single
:class:`mlflow.pyfunc.PythonModel` so the registry tracks one atomic unit.

Loading is the same regardless of where the model came from::

    model = mlflow.pyfunc.load_model("models:/titanic-survival@production")
    preds = model.predict(raw_passenger_df)          # handles preprocessing

On disk the saved artifact keeps the individual files so we can still
`joblib.load` them for debugging if we want.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd


class TitanicSurvivalModel(mlflow.pyfunc.PythonModel):
    """Preprocessor + sklearn estimator as a single pyfunc.

    ``context.artifacts`` must contain ``preprocessor`` and ``estimator``
    paths pointing at joblib pickles produced during training.
    """

    def load_context(self, context) -> None:
        # Imported here so the import survives the MLflow cloudpickle round-trip
        # when ``code_paths`` includes preprocessing.py.
        from preprocessing import Preprocessor

        self.preprocessor = Preprocessor.load(context.artifacts["preprocessor"])
        self.estimator = joblib.load(context.artifacts["estimator"])

    def predict(self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None):
        X = self.preprocessor.transform(model_input)
        return self.estimator.predict(X)


def log_and_register(
    *,
    preprocessor_path: Path,
    estimator_path: Path,
    registered_model_name: str,
    code_paths: list[str] | None = None,
) -> mlflow.models.ModelInfo:
    """Log the pyfunc model to the current MLflow run and register it.

    Returns the :class:`mlflow.models.ModelInfo` so callers can read the
    registered version (``ModelInfo.registered_model_version``).
    """
    return mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=TitanicSurvivalModel(),
        artifacts={
            "preprocessor": str(preprocessor_path),
            "estimator": str(estimator_path),
        },
        code_paths=code_paths or [],
        registered_model_name=registered_model_name,
    )
