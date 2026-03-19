"""
Helpers for selecting which cells to run in an experiment.
"""

from __future__ import annotations

from src.core.types import Cell, CellSelectionConfig


def select_cells(
    all_cells: list[Cell],
    cell_selection: CellSelectionConfig,
) -> list[Cell]:
    """
    Filter the full grid according to the experiment config.
    """
    mode = cell_selection.mode

    if mode == "all":
        return all_cells

    if mode == "include_ids":
        wanted = set(cell_selection.cell_ids or [])
        selected = [cell for cell in all_cells if cell.cell_id in wanted]

        found = {cell.cell_id for cell in selected}
        missing = wanted - found
        if missing:
            raise ValueError(f"Unknown cell IDs requested: {sorted(missing)}")

        return selected

    if mode == "row_col_ranges":
        wanted_rows = set(cell_selection.row_indices or [])
        wanted_cols = set(cell_selection.col_indices or [])

        selected = [
            cell
            for cell in all_cells
            if cell.row in wanted_rows and cell.col in wanted_cols
        ]

        if not selected:
            raise ValueError("cell_selection produced no cells")

        return selected

    raise ValueError(f"Unsupported cell selection mode: {mode}")