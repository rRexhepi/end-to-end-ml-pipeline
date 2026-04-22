"""API tests that train a tiny model + preprocessor into a temp dir
before starting the FastAPI app. Exercises the real lifespan loader,
the filesystem fallback path, and the /metrics endpoint.
"""

from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from monitoring import ReferenceStats
from preprocessing import Preprocessor


@pytest.fixture
def api_client(train_frame, tmp_path, monkeypatch):
    preprocessor = Preprocessor().fit(train_frame)
    X = preprocessor.transform(train_frame)
    model = LogisticRegression(max_iter=200).fit(X, train_frame["Survived"])

    model_path = tmp_path / "model.pkl"
    pp_path = tmp_path / "pp.pkl"
    ref_path = tmp_path / "ref.json"
    joblib.dump(model, model_path)
    preprocessor.save(pp_path)
    ReferenceStats.fit(train_frame).save(ref_path)

    # MODEL_URI unset => filesystem fallback path, which is what the
    # Dockerfile and CI both run with.
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("PREPROCESSOR_PATH", str(pp_path))
    monkeypatch.setenv("REFERENCE_STATS_PATH", str(ref_path))

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


def test_metrics_exposes_prometheus_series(api_client):
    # Prime the monitor with a few predictions so the counters are non-zero.
    api_client.post(
        "/predict",
        json={
            "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
            "Parch": 0, "Fare": 7.25, "Embarked": "S",
        },
    )
    api_client.post(
        "/predict",
        json={
            "Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1,
            "Parch": 0, "Fare": 71.28, "Embarked": "C",
        },
    )

    r = api_client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    text = r.text
    assert "titanic_predictions_total" in text
    assert "titanic_prediction_latency_seconds" in text
    assert 'titanic_input_buffer_size{feature="Age"}' in text
    assert "titanic_model_info" in text
