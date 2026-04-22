"""FastAPI serving layer for the Titanic model.

Loads a `Preprocessor` + sklearn estimator from disk once at startup.
Input validation is done by Pydantic; the Preprocessor handles imputation
with values learned at training time (not inference time).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import Preprocessor

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/random_forest_model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl"))


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


# Loaded in lifespan so unit tests can swap the artifacts by env var.
_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise RuntimeError(
            f"Artifacts missing. Train first: `python src/run_pipeline.py`. "
            f"Looked at {MODEL_PATH} and {PREPROCESSOR_PATH}."
        )
    _state["model"] = joblib.load(MODEL_PATH)
    _state["preprocessor"] = Preprocessor.load(PREPROCESSOR_PATH)
    yield
    _state.clear()


app = FastAPI(title="Titanic Survival API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: Passenger) -> PredictionResponse:
    df = pd.DataFrame([passenger.model_dump()])
    try:
        X = _state["preprocessor"].transform(df)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    pred = int(_state["model"].predict(X)[0])
    return PredictionResponse(prediction=pred, survived=bool(pred))


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(passengers: list[Passenger]) -> list[PredictionResponse]:
    if not passengers:
        raise HTTPException(status_code=400, detail="Empty batch.")
    df = pd.DataFrame([p.model_dump() for p in passengers])
    try:
        X = _state["preprocessor"].transform(df)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    preds = _state["model"].predict(X).tolist()
    return [PredictionResponse(prediction=int(p), survived=bool(p)) for p in preds]
