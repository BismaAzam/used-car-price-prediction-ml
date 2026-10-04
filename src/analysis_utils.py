"""Reusable helpers for the used-car price prediction notebook."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionReport:
    model: str
    mae: float
    rmse: float
    r2: float
    train_seconds: float
    inference_seconds: float

    def as_dict(self) -> dict:
        return {
            "Model": self.model,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "R2": self.r2,
            "Training Time (s)": self.train_seconds,
            "Inference Time (s)": self.inference_seconds,
        }


def evaluate_predictions(name, y_true, y_pred, train_seconds=0.0, inference_seconds=0.0):
    """Create a compact regression evaluation report."""
    return RegressionReport(
        model=name,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        train_seconds=float(train_seconds),
        inference_seconds=float(inference_seconds),
    )


def fit_and_measure(model, x_train, y_train, x_test, y_test, name):
    """Fit a scikit-learn style estimator and measure train/inference time."""
    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test)
    inference_seconds = time.perf_counter() - start

    report = evaluate_predictions(
        name, y_test, predictions, train_seconds, inference_seconds
    )
    return model, predictions, report


def reports_to_frame(reports):
    """Convert RegressionReport objects to a sorted DataFrame."""
    return (
        pd.DataFrame([r.as_dict() for r in reports])
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )
