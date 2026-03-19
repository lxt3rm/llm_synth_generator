"""
Fixed evaluation harness for generated datasets.

This module defines:
- the repeated TSCV splitting protocol
- the x-axis model pipeline
- the y-axis model pipeline

Current convention:
- x_score = mean cross-validated R^2 of KNN regression
- y_score = mean cross-validated R^2 of Linear Regression

Later, the mapping from x/y to algorithms can be changed here without
affecting the rest of the project.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.tscv import make_repeated_splits


def make_splits(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create the fixed repeated TSCV splits used for evaluation.

    Current choice:
    - 5 folds
    - 1 repeats
    - seed 11
    """
    return make_repeated_splits(
        y=y,
        k=5,
        repeats=1,
        seed=11,
    )


def make_x_pipeline() -> Pipeline:
    """
    Create the pipeline used for the x-axis metric.

    Current choice:
    - MinMaxScaler
    - KNeighborsRegressor
    """
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", KNeighborsRegressor()),
    ])


def make_y_pipeline() -> Pipeline:
    """
    Create the pipeline used for the y-axis metric.

    Current choice:
    - MinMaxScaler
    - LinearRegression
    """
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LinearRegression()),
    ])