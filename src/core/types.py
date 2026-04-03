"""
Dataclasses used across the project.

These classes define the main configuration objects and run-time records
used throughout the experiment.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PathsConfig:
    """File and directory paths used by the experiment."""
    accepted_dir: str
    rejected_dir: str
    summaries_dir: str
    logs_dir: str


@dataclass
class GenerationConfig:
    """
    Settings controlling dataset generation and retry behaviour.

    dataset_seed_start is the base seed used to derive deterministic
    per-cell/per-target execution seeds.
    """
    accepted_per_cell: int
    max_retries_per_dataset: int
    max_total_attempts_per_cell: int
    dataset_seed_start: int


@dataclass
class ExecutionConfig:
    """Settings controlling execution of model-generated code."""
    timeout_seconds: int


@dataclass
class CellSelectionConfig:
    """
    Configuration controlling which grid cells should be run.

    Supported modes:
    - all: run every cell in the grid
    - include_ids: run only the listed cell IDs
    - row_col_ranges: run cells whose row and col are both in the given lists
    """
    mode: str
    cell_ids: list[str] | None = None
    row_indices: list[int] | None = None
    col_indices: list[int] | None = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    run_name: str
    paths: PathsConfig
    generation: GenerationConfig
    execution: ExecutionConfig
    cell_selection: CellSelectionConfig


@dataclass
class AxisRange:
    """Numeric range for one performance axis."""
    min: float
    max: float


@dataclass
class GridConfig:
    """Configuration for the 2D performance grid."""
    x: AxisRange
    y: AxisRange
    bins_per_axis: int


@dataclass
class ModelConfig:
    """LLM model configuration."""
    name: str
    temperature: float
    max_output_tokens: int


@dataclass
class Cell:
    """
    One rectangular cell in the 2D performance grid.
    """
    cell_id: str
    row: int
    col: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def center(self) -> tuple[float, float]:
        """Return the center point of the cell."""
        x_center = 0.5 * (self.x_min + self.x_max)
        y_center = 0.5 * (self.y_min + self.y_max)
        return x_center, y_center


@dataclass
class GeneratorResponse:
    """
    Structured response returned by the LLM.
    """
    mechanism_brief: str
    python_code: str
    expected_x_behavior: str
    expected_y_behavior: str
    raw_text: str
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ExecutionResult:
    """
    Result of executing model-generated Python code.
    """
    success: bool
    X: Any | None
    y: Any | None
    error_type: str | None
    error_message: str | None
    runtime_seconds: float


@dataclass
class EvaluationResult:
    """
    Result of evaluating a generated dataset under the fixed harness.
    """
    x_score: float
    y_score: float
    n_rows: int
    n_cols: int
    runtime_seconds: float


@dataclass
class AttemptRecord:
    """
    One logged generation attempt.
    """
    run_id: str
    cell_id: str
    target_dataset_index: int
    attempt_index: int
    execution_seed: int
    accepted: bool
    exhausted: bool
    response_id: str | None
    x_score: float | None
    y_score: float | None
    error_type: str | None
    error_message: str | None
    generator_code_path: str | None
    dataset_csv_path: str | None
    llm_runtime_seconds: float | None
    code_runtime_seconds: float | None
    evaluation_runtime_seconds: float | None
    total_runtime_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass
class AcceptedDatasetRecord:
    """
    Metadata for one accepted dataset.
    """
    run_id: str
    cell_id: str
    accepted_index: int
    attempts_until_hit: int
    x_score: float
    y_score: float
    dataset_csv_path: str
    generator_code_path: str
    metadata_json_path: str


@dataclass
class AttemptOutcome:
    """
    Full in-memory outcome of one attempt run.
    """
    attempt_record: AttemptRecord
    accepted: bool
    llm_response: GeneratorResponse
    execution_result: ExecutionResult
    evaluation_result: EvaluationResult | None
