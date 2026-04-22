"""Tests for the drift + Prometheus monitoring module.

Covers:

* ``ReferenceStats`` fits quantile bins, survives a JSON round-trip, and
  collapses degenerate (constant) features instead of exploding.
* ``compute_psi`` is ~0 on the same distribution and blows up when the
  actual distribution is shifted far from the reference.
* ``DriftMonitor`` wires Prometheus metrics: counters increment by label,
  the latency histogram records, and rendered /metrics output contains
  the expected series names.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from monitoring import (
    DriftMonitor,
    ReferenceStats,
    compute_psi,
    time_block,
)


@pytest.fixture
def reference_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Age": rng.normal(30, 10, size=500).clip(0, 80),
            "Fare": rng.gamma(2.0, 15.0, size=500),
            "SibSp": rng.integers(0, 4, size=500),
            "Parch": rng.integers(0, 3, size=500),
        }
    )


def test_reference_stats_roundtrip(tmp_path, reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    assert set(stats.bin_edges) == {"Age", "Fare", "SibSp", "Parch"}
    for feature, edges in stats.bin_edges.items():
        assert len(edges) >= 3
        counts = stats.bin_counts[feature]
        assert len(counts) == len(edges) - 1
        assert sum(counts) > 0

    path = tmp_path / "ref.json"
    stats.save(path)
    loaded = ReferenceStats.load(path)
    assert loaded.bin_edges == stats.bin_edges
    assert loaded.bin_counts == stats.bin_counts
    assert loaded.n_training_rows == stats.n_training_rows
    # Sanity: file is valid JSON with the expected top-level keys.
    parsed = json.loads(path.read_text())
    assert set(parsed) == {"bin_edges", "bin_counts", "n_training_rows"}


def test_reference_stats_skips_constant_feature():
    df = pd.DataFrame({"Age": [5.0] * 50, "Fare": np.linspace(0, 100, 50)})
    stats = ReferenceStats.fit(df, features=("Age", "Fare"))
    assert "Age" not in stats.bin_edges  # can't bin a constant
    assert "Fare" in stats.bin_edges


def test_psi_zero_for_same_distribution(reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    psi = compute_psi(
        stats.bin_counts["Age"],
        reference_frame["Age"].tolist(),
        stats.bin_edges["Age"],
    )
    assert psi == pytest.approx(0.0, abs=0.05)


def test_psi_flags_shifted_distribution(reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    rng = np.random.default_rng(1)
    shifted = rng.normal(60, 5, size=500).clip(0, 80).tolist()  # training mean was 30
    psi = compute_psi(
        stats.bin_counts["Age"],
        shifted,
        stats.bin_edges["Age"],
    )
    assert psi > 0.25  # "significant drift" rule-of-thumb threshold


def test_drift_monitor_records_and_renders(reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    monitor = DriftMonitor(stats, buffer_size=100)

    inputs = reference_frame.sample(n=20, random_state=0).reset_index(drop=True)
    with time_block() as timer:
        pass
    monitor.record_prediction(inputs, predictions=[0] * 18 + [1] * 2, latency_seconds=timer.elapsed)

    body, content_type = monitor.render()
    text = body.decode()

    assert "text/plain" in content_type
    assert "titanic_predictions_total" in text
    assert 'titanic_predictions_total{label="0"}' in text
    assert 'titanic_predictions_total{label="1"}' in text
    assert "titanic_prediction_latency_seconds" in text
    assert "titanic_feature_drift_psi" in text
    assert 'titanic_input_buffer_size{feature="Age"} 20.0' in text


def test_drift_monitor_without_reference_still_serves_metrics():
    monitor = DriftMonitor(reference=None)
    monitor.record_prediction(
        pd.DataFrame({"Age": [30.0], "Fare": [10.0], "SibSp": [0], "Parch": [0]}),
        predictions=[1],
        latency_seconds=0.01,
    )
    body, _ = monitor.render()
    text = body.decode()
    assert "titanic_predictions_total" in text
    # No reference => no drift gauge values, but the series can still exist empty.
    assert "titanic_feature_drift_psi" not in text or "feature_drift_psi{" not in text.split("HELP titanic_feature_drift_psi")[-1].split("# ")[0]
