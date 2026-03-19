"""
Helpers for creating stable run IDs, dataset IDs, and attempt IDs.
"""

from __future__ import annotations

from datetime import datetime


def make_run_id(run_name: str, now: datetime | None = None) -> str:
    """
    Create a run ID using the run name and current timestamp.

    Example:
        pilot_5x5_v1__20260317_231500
    """
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{run_name}__{timestamp}"


def make_dataset_id(cell_id: str, accepted_index: int) -> str:
    """
    Create a stable ID for one accepted dataset within a cell.

    Example:
        cell_02_03__ds_004
    """
    return f"{cell_id}__ds_{accepted_index:03d}"


def make_attempt_id(cell_id: str, target_dataset_index: int, attempt_index: int) -> str:
    """
    Create a stable ID for one attempt while searching for a dataset.

    Example:
        cell_02_03__target_004__att_002
    """
    return f"{cell_id}__target_{target_dataset_index:03d}__att_{attempt_index:03d}"


def make_generator_filename(attempt_id: str) -> str:
    """
    Filename for saved generator code.
    """
    return f"{attempt_id}.py"


def make_metadata_filename(dataset_id: str) -> str:
    """
    Filename for saved metadata JSON.
    """
    return f"{dataset_id}.json"


def make_csv_filename(dataset_id: str) -> str:
    """
    Filename for saved dataset CSV.
    """
    return f"{dataset_id}.csv"