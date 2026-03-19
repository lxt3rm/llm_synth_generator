"""
Run the generation loop for one target cell.

This module keeps retrying until:
- enough accepted datasets are collected for the cell, or
- the attempt budget is exhausted
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.types import Cell
from src.orchestration.attempt_runner import AttemptRunner


@dataclass
class CellRunSummary:
    """
    Summary of one cell-level generation run.
    """
    cell_id: str
    accepted_count: int
    total_attempts: int
    exhausted: bool


class CellRunner:
    """
    Run the retry loop for one target cell.
    """

    def __init__(
        self,
        *,
        attempt_runner: AttemptRunner,
        accepted_per_cell: int,
        max_retries_per_dataset: int,
        max_total_attempts_per_cell: int,
        bins_per_axis: int,
    ) -> None:
        self.attempt_runner = attempt_runner
        self.accepted_per_cell = accepted_per_cell
        self.max_retries_per_dataset = max_retries_per_dataset
        self.max_total_attempts_per_cell = max_total_attempts_per_cell
        self.bins_per_axis = bins_per_axis

    def run_cell(
        self,
        *,
        run_id: str,
        cell: Cell,
    ) -> CellRunSummary:
        """
        Try to collect accepted datasets for one cell.
        """
        accepted_count = 0
        total_attempts = 0

        is_last_x_bin = (cell.col == self.bins_per_axis - 1)
        is_last_y_bin = (cell.row == self.bins_per_axis - 1)

        while (
            accepted_count < self.accepted_per_cell
            and total_attempts < self.max_total_attempts_per_cell
        ):
            target_dataset_index = accepted_count + 1

            previous_response_id: str | None = None
            previous_x: float | None = None
            previous_y: float | None = None

            hit_this_target = False

            for retry_idx in range(1, self.max_retries_per_dataset + 1):
                if total_attempts >= self.max_total_attempts_per_cell:
                    break

                total_attempts += 1

                if retry_idx == 1:
                    outcome = self.attempt_runner.run_initial_attempt(
                        run_id=run_id,
                        cell=cell,
                        target_dataset_index=target_dataset_index,
                        attempt_index=retry_idx,
                        is_last_x_bin=is_last_x_bin,
                        is_last_y_bin=is_last_y_bin,
                    )
                else:
                    if previous_response_id is None or previous_x is None or previous_y is None:
                        break

                    outcome = self.attempt_runner.run_repair_attempt(
                        run_id=run_id,
                        cell=cell,
                        target_dataset_index=target_dataset_index,
                        attempt_index=retry_idx,
                        achieved_x=previous_x,
                        achieved_y=previous_y,
                        previous_response_id=previous_response_id,
                        is_last_x_bin=is_last_x_bin,
                        is_last_y_bin=is_last_y_bin,
                    )

                previous_response_id = outcome.llm_response.response_id

                if outcome.evaluation_result is not None:
                    previous_x = outcome.evaluation_result.x_score
                    previous_y = outcome.evaluation_result.y_score

                if outcome.accepted:
                    accepted_count += 1
                    hit_this_target = True
                    break

            if not hit_this_target and total_attempts >= self.max_total_attempts_per_cell:
                break

            if not hit_this_target and self.max_retries_per_dataset <= 0:
                break

        exhausted = accepted_count < self.accepted_per_cell
        return CellRunSummary(
            cell_id=cell.cell_id,
            accepted_count=accepted_count,
            total_attempts=total_attempts,
            exhausted=exhausted,
        )