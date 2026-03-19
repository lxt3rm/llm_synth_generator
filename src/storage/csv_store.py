"""
Utilities for saving datasets as CSV files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_dataset_csv(X: np.ndarray, y: np.ndarray, path: str | Path) -> None:
    """
    Save a dataset to CSV with feature columns x1, x2, ..., xd and target y.

    Args:
        X: 2D feature matrix
        y: 1D target vector
        path: output CSV file path
    """
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

    n_cols = X.shape[1]
    feature_names = [f"x{i+1}" for i in range(n_cols)]

    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)