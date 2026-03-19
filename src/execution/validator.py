"""
Validation utilities for generated datasets.

This module checks that the output of a generated function has the expected
structure and contains usable numeric data.
"""

from __future__ import annotations

import numpy as np


def validate_dataset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize a generated dataset.

    Args:
        X: candidate feature matrix
        y: candidate target vector

    Returns:
        A validated pair (X, y) as NumPy arrays.

    Raises:
        ValueError: if the dataset does not satisfy the required structure.
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

    if X.shape[0] == 0:
        raise ValueError("X and y must contain at least one row")

    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature column")

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"X must be numeric, got dtype {X.dtype}")

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"y must be numeric, got dtype {y.dtype}")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")

    return X, y