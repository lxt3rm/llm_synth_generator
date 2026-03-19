"""
Totally Stratified Cross Validation (TSCV) utilities for regression.

This module provides a stratified-like repeated K-fold splitter for regression
by sorting target values and distributing them across folds.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

Split = tuple[np.ndarray, np.ndarray]  # (train_idx, test_idx)


def tscv_split(
    y: np.ndarray,
    n_splits: int,
    random_state: int | None = None,
) -> Iterator[Split]:
    """
    Create one round of stratified-like K-fold splits for a regression target.

    Instances are sorted by target value and distributed across folds while
    balancing fold sizes. Ties among the smallest folds are broken randomly.

    Args:
        y: 1D regression target array of shape (n_instances,)
        n_splits: number of folds
        random_state: seed for reproducibility

    Yields:
        Tuples of (train_idx, test_idx)
    """
    y = np.asarray(y).reshape(-1)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    n_instances = y.shape[0]
    if n_splits > n_instances:
        raise ValueError("n_splits cannot exceed the number of instances")

    rng = np.random.default_rng(seed=random_state)

    indices = np.arange(n_instances)
    sorted_indices = indices[np.argsort(y)]

    folds = [[] for _ in range(n_splits)]
    fold_lengths = np.zeros(n_splits, dtype=int)

    for idx in sorted_indices:
        min_size = fold_lengths.min()
        candidate_folds = np.flatnonzero(fold_lengths == min_size)
        chosen_fold = int(rng.choice(candidate_folds))
        folds[chosen_fold].append(int(idx))
        fold_lengths[chosen_fold] += 1

    test_mask = np.zeros(n_instances, dtype=bool)

    for fold in folds:
        test_mask.fill(False)
        test_mask[fold] = True

        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        yield train_idx, test_idx


def make_repeated_splits(
    y: np.ndarray,
    k: int = 5,
    repeats: int = 2,
    seed: int = 11,
) -> list[Split]:
    """
    Precompute all repeated TSCV splits.

    Args:
        y: 1D regression target array
        k: number of folds
        repeats: number of repeated rounds
        seed: base random seed

    Returns:
        A list of (train_idx, test_idx) splits of length repeats * k
    """
    y = np.asarray(y).reshape(-1)

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    all_splits: list[Split] = []

    for repeat_idx in range(repeats):
        current_seed = seed + repeat_idx
        current_splits = list(
            tscv_split(
                y=y,
                n_splits=k,
                random_state=current_seed,
            )
        )
        all_splits.extend(current_splits)

    return all_splits