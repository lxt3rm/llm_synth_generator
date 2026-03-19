"""
Utilities for acceptance decisions in the 2D performance grid.
"""

from __future__ import annotations

from src.core.types import Cell, EvaluationResult


def is_in_range(value: float, lower: float, upper: float, include_upper: bool) -> bool:
    """
    Check whether a value lies within a numeric interval.

    For non-final bins, we use [lower, upper).
    For the final bin on an axis, we use [lower, upper].
    """
    if include_upper:
        return lower <= value <= upper
    return lower <= value < upper


def is_accepted(
    cell: Cell,
    result: EvaluationResult,
    *,
    is_last_x_bin: bool,
    is_last_y_bin: bool,
) -> bool:
    """
    Check whether an evaluated dataset falls inside the target cell.
    """
    x_ok = is_in_range(result.x_score, cell.x_min, cell.x_max, include_upper=is_last_x_bin)
    y_ok = is_in_range(result.y_score, cell.y_min, cell.y_max, include_upper=is_last_y_bin)
    return x_ok and y_ok