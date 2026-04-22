"""Online drift monitoring and Prometheus metrics for the Titanic API.

The story this module tells:

1. **At training time** we snapshot a reference distribution for every
   numeric feature we care about — decile bin edges and per-bin counts.
   Saved as JSON next to the model artifacts.
2. **At serving time** we keep a bounded ring buffer of recent inputs.
   On every Prometheus scrape we re-bin the buffer against the reference
   edges and compute **PSI** (Population Stability Index) per feature,
   then expose it as a gauge.
3. **Rule of thumb**: ``PSI < 0.1`` → no drift, ``0.1 ≤ PSI < 0.25`` →
   moderate drift worth investigating, ``PSI ≥ 0.25`` → significant drift,
   model should probably be retrained.

Why PSI and not Evidently / WhyLogs? Evidently is the right call when you
need a full report with HTML, statistical test batteries, and categorical
drift out of the box. For an always-on ``/metrics`` endpoint backing a
Grafana dashboard, a single scalar per feature is what you actually plot,
and PSI is the standard for that. One dep less, and the code shows I
understand what "drift" means rather than just wiring a black box.

On a static Kaggle dataset the drift numbers will be ~0 by construction —
the point is the *wiring*. Pointed at a live stream, the same code earns
its keep.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

MONITORED_FEATURES: tuple[str, ...] = ("Age", "Fare", "SibSp", "Parch")
DEFAULT_BUFFER_SIZE = 1000
DEFAULT_BIN_COUNT = 10
_PSI_EPSILON = 1e-6  # avoid log(0) when a bin empties out


@dataclass(frozen=True)
class ReferenceStats:
    """Per-feature bin edges + counts captured on the training set.

    ``bin_edges`` come from ``np.quantile`` at evenly spaced probabilities
    so each reference bin holds roughly the same share of training mass.
    """

    bin_edges: dict[str, list[float]]
    bin_counts: dict[str, list[int]]
    n_training_rows: int

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        features: Iterable[str] = MONITORED_FEATURES,
        bins: int = DEFAULT_BIN_COUNT,
    ) -> ReferenceStats:
        edges: dict[str, list[float]] = {}
        counts: dict[str, list[int]] = {}
        for feature in features:
            if feature not in df.columns:
                continue
            col = df[feature].dropna().astype(float)
            if col.empty:
                continue
            qs = np.linspace(0, 1, bins + 1)
            raw_edges = np.quantile(col, qs).tolist()
            # Collapse degenerate bins (can happen when a feature has ties)
            # so np.digitize doesn't create zero-width intervals.
            deduped = [raw_edges[0]]
            for edge in raw_edges[1:]:
                if edge > deduped[-1]:
                    deduped.append(edge)
            if len(deduped) < 3:
                # Feature too narrow to bin meaningfully; skip it.
                continue
            edges[feature] = deduped
            hist, _ = np.histogram(col, bins=deduped)
            counts[feature] = hist.astype(int).tolist()
        return cls(bin_edges=edges, bin_counts=counts, n_training_rows=len(df))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "bin_edges": self.bin_edges,
                    "bin_counts": self.bin_counts,
                    "n_training_rows": self.n_training_rows,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> ReferenceStats:
        data = json.loads(Path(path).read_text())
        return cls(
            bin_edges={k: list(map(float, v)) for k, v in data["bin_edges"].items()},
            bin_counts={k: list(map(int, v)) for k, v in data["bin_counts"].items()},
            n_training_rows=int(data["n_training_rows"]),
        )


def compute_psi(
    reference_counts: list[int],
    actual_values: list[float],
    bin_edges: list[float],
) -> float:
    """Population Stability Index of ``actual_values`` vs. the reference bins.

    Returns ``NaN`` if either side has zero mass (meaning we can't compare).
    """
    if not actual_values or sum(reference_counts) == 0:
        return float("nan")
    expected = np.array(reference_counts, dtype=float)
    expected_pct = expected / expected.sum()
    actual, _ = np.histogram(np.asarray(actual_values, dtype=float), bins=bin_edges)
    if actual.sum() == 0:
        return float("nan")
    actual_pct = actual.astype(float) / actual.sum()
    # Clamp zeros so log is finite; acts like Laplace smoothing at the bin level.
    expected_pct = np.clip(expected_pct, _PSI_EPSILON, None)
    actual_pct = np.clip(actual_pct, _PSI_EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


class DriftMonitor:
    """Bounded ring buffer of recent inputs + Prometheus-exposed PSI gauges.

    The monitor is thread-safe because FastAPI workers may share it (though
    in uvicorn-default single-worker mode the lock is uncontended). It owns
    its own ``CollectorRegistry`` so the test suite can instantiate monitors
    without polluting the process-global registry.
    """

    def __init__(
        self,
        reference: ReferenceStats | None,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        namespace: str = "titanic",
        registry: CollectorRegistry | None = None,
    ):
        self._reference = reference
        self._buffers: dict[str, deque[float]] = {
            f: deque(maxlen=buffer_size) for f in MONITORED_FEATURES
        }
        self._lock = threading.Lock()
        self._registry = registry or CollectorRegistry()

        self.predictions_total = Counter(
            f"{namespace}_predictions_total",
            "Predictions served, labelled by output class.",
            labelnames=("label",),
            registry=self._registry,
        )
        self.prediction_latency = Histogram(
            f"{namespace}_prediction_latency_seconds",
            "End-to-end latency of a /predict call (seconds).",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self._registry,
        )
        self.feature_drift_psi = Gauge(
            f"{namespace}_feature_drift_psi",
            "PSI of recent inputs vs. the training distribution.",
            labelnames=("feature",),
            registry=self._registry,
        )
        self.buffer_size = Gauge(
            f"{namespace}_input_buffer_size",
            "Rows currently held for drift scoring.",
            labelnames=("feature",),
            registry=self._registry,
        )
        self.model_info = Info(
            f"{namespace}_model",
            "Metadata about the served model.",
            registry=self._registry,
        )

    # --- instrumentation --------------------------------------------------
    def record_prediction(self, inputs: pd.DataFrame, predictions: Iterable[int], latency_seconds: float) -> None:
        with self._lock:
            for feature, buf in self._buffers.items():
                if feature in inputs.columns:
                    for value in inputs[feature].tolist():
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            continue
                        buf.append(float(value))
        for label in predictions:
            self.predictions_total.labels(label=str(int(label))).inc()
        self.prediction_latency.observe(latency_seconds)

    def set_model_info(self, **fields: str) -> None:
        # prometheus_client Info requires str values.
        self.model_info.info({k: str(v) for k, v in fields.items()})

    # --- scraping ---------------------------------------------------------
    def _refresh_drift_gauges(self) -> None:
        if self._reference is None:
            return
        with self._lock:
            snapshot = {f: list(buf) for f, buf in self._buffers.items()}
        for feature, values in snapshot.items():
            self.buffer_size.labels(feature=feature).set(len(values))
            if feature not in self._reference.bin_edges:
                continue
            psi = compute_psi(
                self._reference.bin_counts[feature],
                values,
                self._reference.bin_edges[feature],
            )
            if not math.isnan(psi):
                self.feature_drift_psi.labels(feature=feature).set(psi)

    def render(self) -> tuple[bytes, str]:
        """Return (body, content_type) suitable for a Prometheus /metrics response."""
        self._refresh_drift_gauges()
        return generate_latest(self._registry), CONTENT_TYPE_LATEST


class _LatencyTimer:
    """Tiny helper so instrumentation at the call site reads as one line."""

    def __enter__(self) -> _LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def time_block() -> _LatencyTimer:
    return _LatencyTimer()
