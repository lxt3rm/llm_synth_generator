"""
Prompt-building utilities for the synthetic dataset generator.
"""

from __future__ import annotations

from pathlib import Path

from src.core.types import Cell


def _read_text_file(path: str | Path) -> str:
    """Read a UTF-8 text file and return its contents."""
    return Path(path).read_text(encoding="utf-8")


class PromptBuilder:
    """
    Builds prompts for initial generation and repair generation.

    Templates are stored in prompts/ so they can be edited without changing code.
    """

    def __init__(
        self,
        system_prompt_path: str = "prompts/system_prompt.txt",
        initial_prompt_path: str = "prompts/initial_prompt.txt",
        repair_prompt_path: str = "prompts/repair_prompt.txt",
    ) -> None:
        self.system_template = _read_text_file(system_prompt_path)
        self.initial_template = _read_text_file(initial_prompt_path)
        self.repair_template = _read_text_file(repair_prompt_path)

    def build_system_prompt(self) -> str:
        """Return the system prompt text."""
        return self.system_template

    def build_initial_prompt(self, cell: Cell) -> str:
        """
        Build the initial generation prompt for a target cell.
        """
        target_description = (
            f"Target x-range: [{cell.x_min:.6f}, {cell.x_max:.6f}]\n"
            f"Target y-range: [{cell.y_min:.6f}, {cell.y_max:.6f}]\n"
            f"Target center: ({cell.center()[0]:.6f}, {cell.center()[1]:.6f})\n"
        )
        return self.initial_template.format(target_description=target_description)

    def build_repair_prompt(
        self,
        cell: Cell,
        achieved_x: float,
        achieved_y: float,
    ) -> str:
        """
        Build a repair prompt after a miss.
        """
        target_description = (
            f"Target x-range: [{cell.x_min:.6f}, {cell.x_max:.6f}]\n"
            f"Target y-range: [{cell.y_min:.6f}, {cell.y_max:.6f}]\n"
            f"Target center: ({cell.center()[0]:.6f}, {cell.center()[1]:.6f})\n"
        )

        achieved_description = (
            f"Achieved x-score: {achieved_x:.6f}\n"
            f"Achieved y-score: {achieved_y:.6f}\n"
        )

        return self.repair_template.format(
            target_description=target_description,
            achieved_description=achieved_description,
        )