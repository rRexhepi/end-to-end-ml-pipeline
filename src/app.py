"""FastAPI serving layer for the Titanic model.

Two loading paths, picked in order:

1. **Model Registry URI** (``MODEL_URI``, e.g. ``models:/titanic-survival@production``).
   This is the production path — one atomic pyfunc artifact, versioned, promotable
   via ``mlflow.MlflowClient.set_registered_model_alias``.
2. **Filesystem pickles** (``MODEL_PATH`` + ``PREPROCESSOR_PATH``). Used by CI,
   the Dockerfile's baked-in artifacts, and local dev before a Registry exists.

Both paths expose the same ``predict(df) -> np.ndarray`` interface so the rest
of the app doesn't care which one loaded the model.

The app also serves Prometheus ``/metrics``: per-class prediction counters,
prediction latency histogram, and PSI-based feature drift gauges. See
``src/monitoring.py`` for the drift story.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Protocol

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from monitoring import DriftMonitor, ReferenceStats, time_block
from preprocessing import Preprocessor

logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI")  # e.g. "models:/titanic-survival@production"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/random_forest_model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl"))
REFERENCE_STATS_PATH = Path(
    os.getenv("REFERENCE_STATS_PATH", "models/reference_stats.json")
)


class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float | None = Field(default=None, ge=0)
    Embarked: Literal["S", "C", "Q"] = "S"


class PredictionResponse(BaseModel):
    prediction: int
    survived: bool


class _Predictor(Protocol):
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


class _FilesystemPredictor:
    """Legacy two-pickle loader wrapped in the pyfunc-style interface."""

    def __init__(self, estimator_path: Path, preprocessor_path: Path):
        self._estimator = joblib.load(estimator_path)
        self._preprocessor = Preprocessor.load(preprocessor_path)
        self.source = f"filesystem:{estimator_path}"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._preprocessor.transform(df)
        return np.asarray(self._estimator.predict(X))


class _RegistryPredictor:
    """Thin wrapper so ``pyfunc.load_model`` matches the _FilesystemPredictor API."""

    def __init__(self, uri: str):
        self._model = mlflow.pyfunc.load_model(uri)
        self.source = f"registry:{uri}"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(df))


def _load_predictor() -> _Predictor:
    if MODEL_URI:
        logger.info("Loading model from registry URI: %s", MODEL_URI)
        return _RegistryPredictor(MODEL_URI)
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise RuntimeError(
            "No MODEL_URI set and filesystem artifacts missing. "
            "Train first (`python src/train_model.py`) or set MODEL_URI to a "
            f"registry URI. Looked at {MODEL_PATH} and {PREPROCESSOR_PATH}."
        )
    logger.info("Loading model from filesystem: %s", MODEL_PATH)
    return _FilesystemPredictor(MODEL_PATH, PREPROCESSOR_PATH)


def _load_reference() -> ReferenceStats | None:
    if REFERENCE_STATS_PATH.exists():
        return ReferenceStats.load(REFERENCE_STATS_PATH)
    logger.warning(
        "Reference stats missing at %s; drift gauges will be empty.",
        REFERENCE_STATS_PATH,
    )
    return None


_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor = _load_predictor()
    reference = _load_reference()
    monitor = DriftMonitor(reference)
    monitor.set_model_info(
        source=predictor.source,
        uri=MODEL_URI or "",
        path=str(MODEL_PATH) if not MODEL_URI else "",
    )
    _state["predictor"] = predictor
    _state["monitor"] = monitor
    yield
    _state.clear()


app = FastAPI(title="Titanic Survival API", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _predict_frame(df: pd.DataFrame) -> np.ndarray:
    predictor: _Predictor = _state["predictor"]
    monitor: DriftMonitor = _state["monitor"]
    try:
        with time_block() as timer:
            preds = predictor.predict(df)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    monitor.record_prediction(df, preds.tolist(), timer.elapsed)
    return preds


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: Passenger) -> PredictionResponse:
    df = pd.DataFrame([passenger.model_dump()])
    preds = _predict_frame(df)
    pred = int(preds[0])
    return PredictionResponse(prediction=pred, survived=bool(pred))


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(passengers: list[Passenger]) -> list[PredictionResponse]:
    if not passengers:
        raise HTTPException(status_code=400, detail="Empty batch.")
    df = pd.DataFrame([p.model_dump() for p in passengers])
    preds = _predict_frame(df).tolist()
    return [PredictionResponse(prediction=int(p), survived=bool(p)) for p in preds]


@app.get("/metrics")
def metrics() -> Response:
    monitor: DriftMonitor = _state["monitor"]
    body, content_type = monitor.render()
    return Response(content=body, media_type=content_type)
