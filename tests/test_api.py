"""API tests that train a tiny model + preprocessor into a temp dir
before starting the FastAPI app. This exercises the real lifespan loader.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from preprocessing import Preprocessor


@pytest.fixture
def api_client(train_frame, tmp_path, monkeypatch):
    preprocessor = Preprocessor().fit(train_frame)
    X = preprocessor.transform(train_frame)
    model = LogisticRegression(max_iter=200).fit(X, train_frame["Survived"])

    model_path = tmp_path / "model.pkl"
    pp_path = tmp_path / "pp.pkl"
    joblib.dump(model, model_path)
    preprocessor.save(pp_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("PREPROCESSOR_PATH", str(pp_path))

    # Import after env is set so the module-level constants pick it up.
    import importlib
    import app as app_module
    importlib.reload(app_module)

    with TestClient(app_module.app) as client:
        yield client


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid_payload(api_client):
    r = api_client.post(
        "/predict",
        json={
            "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
            "Parch": 0, "Fare": 7.25, "Embarked": "S",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert isinstance(body["survived"], bool)


def test_predict_rejects_bad_sex(api_client):
    r = api_client.post(
        "/predict",
        json={
            "Pclass": 3, "Sex": "other", "Age": 22, "SibSp": 1,
            "Parch": 0, "Fare": 7.25, "Embarked": "S",
        },
    )
    # Pydantic literal validation returns 422.
    assert r.status_code == 422


def test_predict_missing_field(api_client):
    r = api_client.post("/predict", json={"Pclass": 3, "Sex": "male"})
    assert r.status_code == 422


def test_predict_batch(api_client):
    r = api_client.post(
        "/predict/batch",
        json=[
            {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
            {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C"},
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2


def test_predict_batch_empty_rejected(api_client):
    r = api_client.post("/predict/batch", json=[])
    assert r.status_code == 400
