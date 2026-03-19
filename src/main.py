"""
Main entry point for the project.

This step:
- runs the full experiment across the selected grid cells
- saves the run summary
- computes a post-run summary from attempts.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.summarise_attempts import save_attempt_summary
from src.core.cell_selection import select_cells
from src.core.config import (
    load_experiment_config,
    load_grid_config,
    load_model_config,
)
from src.core.grid import build_grid
from src.core.ids import make_run_id
from src.evaluation.evaluator import DatasetEvaluator
from src.execution.code_runner import CodeRunner
from src.llm.openai_client import OpenAIGeneratorClient
from src.llm.prompt_builder import PromptBuilder
from src.orchestration.attempt_runner import AttemptRunner
from src.orchestration.cell_runner import CellRunner
from src.orchestration.experiment_runner import ExperimentRunner
from src.storage.log_store import JsonlLogStore
from src.storage.path_manager import PathManager


def main() -> None:
    """
    Run the full experiment across the selected cells and summarise the attempt log.
    """
    experiment_config = load_experiment_config("configs/experiment.yaml")
    grid_config = load_grid_config("configs/grid.yaml")
    model_config = load_model_config("configs/model.yaml")

    all_cells = build_grid(grid_config)
    cells = select_cells(all_cells, experiment_config.cell_selection)
    run_id = make_run_id(experiment_config.run_name)

    path_manager = PathManager(experiment_config)
    path_manager.ensure_base_directories()

    prompt_builder = PromptBuilder()
    llm_client = OpenAIGeneratorClient(model_config)
    code_runner = CodeRunner(
        timeout_seconds=experiment_config.execution.timeout_seconds,
    )
    evaluator = DatasetEvaluator()
    log_store = JsonlLogStore()

    attempt_runner = AttemptRunner(
        experiment_config=experiment_config,
        model_config=model_config,
        path_manager=path_manager,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        code_runner=code_runner,
        evaluator=evaluator,
        log_store=log_store,
    )

    cell_runner = CellRunner(
        attempt_runner=attempt_runner,
        accepted_per_cell=experiment_config.generation.accepted_per_cell,
        max_retries_per_dataset=experiment_config.generation.max_retries_per_dataset,
        max_total_attempts_per_cell=experiment_config.generation.max_total_attempts_per_cell,
        bins_per_axis=grid_config.bins_per_axis,
    )

    experiment_runner = ExperimentRunner(
        cell_runner=cell_runner,
        path_manager=path_manager,
    )

    print("Cell selection:")
    print(f"  Mode: {experiment_config.cell_selection.mode}")
    print(f"  Selected cells: {len(cells)} / {len(all_cells)}")
    print(f"  Cell IDs: {[cell.cell_id for cell in cells]}")
    print()

    run_summary = experiment_runner.run(
        run_id=run_id,
        cells=cells,
    )

    attempt_summary_path = Path(experiment_config.paths.logs_dir) / "attempt_analysis.json"
    attempt_summary = save_attempt_summary(
        attempt_log_path=path_manager.get_attempts_log_path(),
        output_path=attempt_summary_path,
    )

    print("Full experiment completed.")
    print(f"Run ID: {run_id}")
    print(f"Total selected cells: {len(cells)}")
    print(f"Total accepted: {run_summary['total_accepted']}")
    print(f"Total attempts: {run_summary['total_attempts']}")
    print(f"Overall hit rate: {run_summary['overall_hit_rate']:.4f}")
    print()
    print("Post-run attempt analysis:")
    print(f"Mean attempts per accepted dataset: {attempt_summary['mean_attempts_per_accepted_dataset']}")
    print(f"Mean total runtime per attempt: {attempt_summary['mean_total_runtime_seconds']}")
    print(f"Sum total tokens: {attempt_summary['sum_total_tokens']}")
    print(f"Mean total tokens per attempt: {attempt_summary['mean_total_tokens_per_attempt']}")
    print()
    print(f"Attempts log: {path_manager.get_attempts_log_path()}")
    print(f"Run summary: {path_manager.get_summary_path()}")
    print(f"Attempt analysis: {attempt_summary_path}")


if __name__ == "__main__":
    main()