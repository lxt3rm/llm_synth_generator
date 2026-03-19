"""
Dataset evaluator.

Given X and y, this module computes the dataset's coordinates in the 2D
performance space under the fixed evaluation harness.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import cross_val_score

from src.core.types import EvaluationResult
from src.evaluation.harness import make_splits, make_x_pipeline, make_y_pipeline


class DatasetEvaluator:
    """
    Evaluates a regression dataset under the fixed harness.

    Current convention:
    - x_score = mean R^2 of the x-axis pipeline
    - y_score = mean R^2 of the y-axis pipeline
    """

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> EvaluationResult:
        """
        Evaluate the dataset and return its 2D performance coordinates.

        Args:
            X: 2D feature matrix of shape (n_rows, n_cols)
            y: 1D target vector of shape (n_rows,)

        Returns:
            EvaluationResult containing x_score, y_score, dataset shape,
            and evaluation runtime.
        """
        start_time = time.perf_counter()

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, got {X.shape[0]} and {y.shape[0]}"
            )

        splits = make_splits(y)

        x_pipeline = make_x_pipeline()
        y_pipeline = make_y_pipeline()

        x_scores = cross_val_score(
            estimator=x_pipeline,
            X=X,
            y=y,
            cv=splits,
            scoring="r2",
        )

        y_scores = cross_val_score(
            estimator=y_pipeline,
            X=X,
            y=y,
            cv=splits,
            scoring="r2",
        )

        runtime_seconds = time.perf_counter() - start_time

        return EvaluationResult(
            x_score=float(np.mean(x_scores)),
            y_score=float(np.mean(y_scores)),
            n_rows=int(X.shape[0]),
            n_cols=int(X.shape[1]),
            runtime_seconds=runtime_seconds,
        )

    def evaluate_to_dict(self, X: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
        """
        Evaluate a dataset and return a plain dictionary.

        This helper is useful for exposing evaluation to generated code
        running inside CodeRunner.
        """
        result = self.evaluate(X, y)
        return {
            "x_score": result.x_score,
            "y_score": result.y_score,
            "n_rows": result.n_rows,
            "n_cols": result.n_cols,
            "runtime_seconds": result.runtime_seconds,
        }