"""
Helpers for building output paths for accepted datasets, rejected attempts,
metadata files, and logs.
"""

from __future__ import annotations

from pathlib import Path

from src.core.types import ExperimentConfig


class PathManager:
    """
    Centralizes all file and directory path creation.

    This keeps path logic out of the orchestration code later.
    """

    def __init__(self, experiment_config: ExperimentConfig) -> None:
        self.config = experiment_config

    def ensure_base_directories(self) -> None:
        """Create the main output directories if they do not already exist."""
        Path(self.config.paths.accepted_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.paths.rejected_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.paths.summaries_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.paths.logs_dir).mkdir(parents=True, exist_ok=True)

    def get_cell_accepted_dir(self, cell_id: str) -> Path:
        """Directory for accepted datasets belonging to one cell."""
        path = Path(self.config.paths.accepted_dir) / cell_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cell_rejected_dir(self, cell_id: str) -> Path:
        """Directory for rejected attempts belonging to one cell."""
        path = Path(self.config.paths.rejected_dir) / cell_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_dataset_dir(self, cell_id: str, dataset_id: str) -> Path:
        """Directory for one accepted dataset and its artifacts."""
        path = self.get_cell_accepted_dir(cell_id) / dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_attempt_dir(self, cell_id: str, attempt_id: str) -> Path:
        """Directory for one rejected attempt and its artifacts."""
        path = self.get_cell_rejected_dir(cell_id) / attempt_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_attempts_log_path(self) -> Path:
        """Path to the full attempt log."""
        return Path(self.config.paths.logs_dir) / "attempts.jsonl"

    def get_accepted_log_path(self) -> Path:
        """Path to the accepted-datasets log."""
        return Path(self.config.paths.logs_dir) / "accepted.jsonl"

    def get_failures_log_path(self) -> Path:
        """Path to the execution-failures log."""
        return Path(self.config.paths.logs_dir) / "failures.jsonl"

    def get_summary_path(self) -> Path:
        """Path to the final run summary."""
        return Path(self.config.paths.logs_dir) / "run_summary.json"