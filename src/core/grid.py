"""
Utilities for building the 2D performance grid.
"""

from __future__ import annotations

from src.core.types import Cell, GridConfig


def build_grid(grid_config: GridConfig) -> list[Cell]:
    """
    Build a rectangular grid of cells over the performance space.

    The grid is defined by:
    - x range [x.min, x.max]
    - y range [y.min, y.max]
    - bins_per_axis bins on each axis

    Returns:
        A list of Cell objects in row-major order.
        Rows correspond to y bins and columns correspond to x bins.
    """
    bins = grid_config.bins_per_axis

    x_width = (grid_config.x.max - grid_config.x.min) / bins
    y_width = (grid_config.y.max - grid_config.y.min) / bins

    cells: list[Cell] = []

    for row in range(bins):
        y_min = grid_config.y.min + row * y_width
        y_max = grid_config.y.min + (row + 1) * y_width

        for col in range(bins):
            x_min = grid_config.x.min + col * x_width
            x_max = grid_config.x.min + (col + 1) * x_width

            cell = Cell(
                cell_id=f"cell_{row:02d}_{col:02d}",
                row=row,
                col=col,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            cells.append(cell)

    return cells