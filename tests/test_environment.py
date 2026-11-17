"""Dependency integration checks, not tests of shipment prediction accuracy."""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def test_utc_arithmetic_across_a_day_boundary():
    arrival = pd.Timestamp("2025-01-01T23:00:00Z")
    delivery = arrival + pd.Timedelta(hours=7.5)
    assert delivery == pd.Timestamp("2025-01-02T06:30:00Z")
    assert (delivery - arrival).total_seconds() / 3600 == 7.5


def test_tabular_pipeline_prediction_and_serialization(tmp_path):
    training = pd.DataFrame({
        "hours": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        "route": ["A", "B", "A", "B", "A", "B"],
    })
    target = np.array([5.0, 8.0, 9.0, 12.0, 13.0, 16.0])
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), ["hours"]),
        ("route", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["route"]),
    ])
    pipeline = Pipeline([("features", preprocessing), ("regressor", Ridge(alpha=1.0))])
    pipeline.fit(training, target)
    unseen = pd.DataFrame({"hours": [3.0, 7.0], "route": ["A", "NEW"]})
    before = pipeline.predict(unseen)
    assert before.shape == (2,)
    assert np.isfinite(before).all()
    destination = tmp_path / "toy_pipeline.joblib"
    joblib.dump(pipeline, destination)
    # Load only the artifact created by this test, never an untrusted model file.
    after = joblib.load(destination).predict(unseen)
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)


def test_headless_png_rendering():
    figure = Figure(figsize=(3, 2))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot([0, 1, 2], [0, 2, 1])
    axes.set_xlabel("Toy input")
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(buffer.getvalue()) > 100
