"""
Utilities for loading YAML configuration files into typed dataclasses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.types import (
    AxisRange,
    CellSelectionConfig,
    ExecutionConfig,
    ExperimentConfig,
    GenerationConfig,
    GridConfig,
    ModelConfig,
    PathsConfig,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the YAML content is empty or invalid.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {file_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {file_path}")

    return data


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    """
    Fetch a required key from a dictionary.

    Raises:
        KeyError: if the key is missing.
    """
    if key not in mapping:
        raise KeyError(f"Missing required key '{key}' in {context}")
    return mapping[key]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the experiment configuration from YAML."""
    data = _read_yaml(path)

    run_name = _require(data, "run_name", "experiment config")

    paths_data = _require(data, "paths", "experiment config")
    generation_data = _require(data, "generation", "experiment config")
    execution_data = _require(data, "execution", "experiment config")
    cell_selection_data = data.get("cell_selection", {"mode": "all"})

    paths = PathsConfig(
        accepted_dir=_require(paths_data, "accepted_dir", "experiment.paths"),
        rejected_dir=_require(paths_data, "rejected_dir", "experiment.paths"),
        summaries_dir=_require(paths_data, "summaries_dir", "experiment.paths"),
        logs_dir=_require(paths_data, "logs_dir", "experiment.paths"),
    )

    generation = GenerationConfig(
        accepted_per_cell=int(_require(generation_data, "accepted_per_cell", "experiment.generation")),
        max_retries_per_dataset=int(_require(generation_data, "max_retries_per_dataset", "experiment.generation")),
        max_total_attempts_per_cell=int(_require(generation_data, "max_total_attempts_per_cell", "experiment.generation")),
        dataset_seed_start=int(_require(generation_data, "dataset_seed_start", "experiment.generation")),
    )

    execution = ExecutionConfig(
        timeout_seconds=int(_require(execution_data, "timeout_seconds", "experiment.execution")),
    )

    cell_selection = CellSelectionConfig(
        mode=str(_require(cell_selection_data, "mode", "experiment.cell_selection")),
        cell_ids=cell_selection_data.get("cell_ids"),
        row_indices=cell_selection_data.get("row_indices"),
        col_indices=cell_selection_data.get("col_indices"),
    )

    _validate_experiment_config(run_name, generation, execution, cell_selection)

    return ExperimentConfig(
        run_name=str(run_name),
        paths=paths,
        generation=generation,
        execution=execution,
        cell_selection=cell_selection,
    )


def load_grid_config(path: str | Path) -> GridConfig:
    """Load the grid configuration from YAML."""
    data = _read_yaml(path)

    x_data = _require(data, "x", "grid config")
    y_data = _require(data, "y", "grid config")
    bins_per_axis = int(_require(data, "bins_per_axis", "grid config"))

    x = AxisRange(
        min=float(_require(x_data, "min", "grid.x")),
        max=float(_require(x_data, "max", "grid.x")),
    )

    y = AxisRange(
        min=float(_require(y_data, "min", "grid.y")),
        max=float(_require(y_data, "max", "grid.y")),
    )

    _validate_grid_config(x, y, bins_per_axis)

    return GridConfig(
        x=x,
        y=y,
        bins_per_axis=bins_per_axis,
    )


def load_model_config(path: str | Path) -> ModelConfig:
    """Load the model configuration from YAML."""
    data = _read_yaml(path)

    model_data = _require(data, "model", "model config")

    name = str(_require(model_data, "name", "model.name"))
    temperature = float(_require(model_data, "temperature", "model.temperature"))
    max_output_tokens = int(_require(model_data, "max_output_tokens", "model.max_output_tokens"))

    _validate_model_config(name, temperature, max_output_tokens)

    return ModelConfig(
        name=name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _validate_experiment_config(
    run_name: str,
    generation: GenerationConfig,
    execution: ExecutionConfig,
    cell_selection: CellSelectionConfig,
) -> None:
    """Validate basic experiment settings."""
    if not str(run_name).strip():
        raise ValueError("run_name must be a non-empty string")

    if generation.accepted_per_cell <= 0:
        raise ValueError("accepted_per_cell must be > 0")

    if generation.max_retries_per_dataset <= 0:
        raise ValueError("max_retries_per_dataset must be > 0")

    if generation.max_total_attempts_per_cell <= 0:
        raise ValueError("max_total_attempts_per_cell must be > 0")

    if execution.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    _validate_cell_selection_config(cell_selection)


def _validate_cell_selection_config(cell_selection: CellSelectionConfig) -> None:
    """Validate cell-selection settings."""
    allowed_modes = {"all", "include_ids", "row_col_ranges"}

    if cell_selection.mode not in allowed_modes:
        raise ValueError(
            f"cell_selection.mode must be one of {allowed_modes}, got {cell_selection.mode}"
        )

    if cell_selection.mode == "include_ids":
        if not cell_selection.cell_ids:
            raise ValueError("cell_ids must be provided when mode='include_ids'")

    if cell_selection.mode == "row_col_ranges":
        if cell_selection.row_indices is None or cell_selection.col_indices is None:
            raise ValueError(
                "row_indices and col_indices must be provided when mode='row_col_ranges'"
            )


def _validate_grid_config(x: AxisRange, y: AxisRange, bins_per_axis: int) -> None:
    """Validate grid settings."""
    if x.min >= x.max:
        raise ValueError("x.min must be less than x.max")

    if y.min >= y.max:
        raise ValueError("y.min must be less than y.max")

    if bins_per_axis <= 0:
        raise ValueError("bins_per_axis must be > 0")


def _validate_model_config(name: str, temperature: float, max_output_tokens: int) -> None:
    """Validate model settings."""
    if not name.strip():
        raise ValueError("model name must be a non-empty string")

    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be > 0")