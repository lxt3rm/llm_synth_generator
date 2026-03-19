"""
Run the full experiment across all cells in the 2D grid.

This module coordinates:
- iterating over all cells
- calling CellRunner for each cell
- collecting per-cell summaries
- computing overall totals
- saving a final run summary JSON
"""

from __future__ import annotations

from dataclasses import asdict

from src.core.types import Cell
from src.orchestration.cell_runner import CellRunSummary, CellRunner
from src.storage.json_store import save_json
from src.storage.path_manager import PathManager


class ExperimentRunner:
    """
    Run the full experiment over all target cells.
    """

    def __init__(
        self,
        *,
        cell_runner: CellRunner,
        path_manager: PathManager,
    ) -> None:
        self.cell_runner = cell_runner
        self.path_manager = path_manager

    def run(
        self,
        *,
        run_id: str,
        cells: list[Cell],
    ) -> dict:
        """
        Run the full experiment over every cell.

        Args:
            run_id: unique identifier for this experiment run
            cells: list of grid cells to process

        Returns:
            A summary dictionary containing per-cell and overall results.
        """
        cell_summaries: list[CellRunSummary] = []

        print("Starting full experiment run.")
        print(f"Total cells to process: {len(cells)}")
        print()

        for cell_idx, cell in enumerate(cells, start=1):
            print(f"[{cell_idx}/{len(cells)}] Running cell {cell.cell_id} ...")

            summary = self.cell_runner.run_cell(
                run_id=run_id,
                cell=cell,
            )
            cell_summaries.append(summary)

            print(
                f"  accepted={summary.accepted_count}, "
                f"attempts={summary.total_attempts}, "
                f"exhausted={summary.exhausted}"
            )
            print()

        total_accepted = sum(summary.accepted_count for summary in cell_summaries)
        total_attempts = sum(summary.total_attempts for summary in cell_summaries)
        exhausted_cells = sum(1 for summary in cell_summaries if summary.exhausted)

        overall_hit_rate = (
            total_accepted / total_attempts if total_attempts > 0 else 0.0
        )

        run_summary = {
            "run_id": run_id,
            "n_cells": len(cells),
            "total_accepted": total_accepted,
            "total_attempts": total_attempts,
            "overall_hit_rate": overall_hit_rate,
            "exhausted_cells": exhausted_cells,
            "cell_summaries": [asdict(summary) for summary in cell_summaries],
        }

        summary_path = self.path_manager.get_summary_path()
        save_json(run_summary, summary_path)

        return run_summary