"""
Run one end-to-end generation attempt.

Flow:
- build prompt
- call LLM
- execute returned code
- evaluate dataset
- decide acceptance
- save artifacts
- log attempt
"""

from __future__ import annotations

from src.core.ids import (
    make_attempt_id,
    make_csv_filename,
    make_dataset_id,
    make_generator_filename,
    make_metadata_filename,
)
from src.core.types import (
    AttemptOutcome,
    AttemptRecord,
    Cell,
    ExperimentConfig,
    ModelConfig,
)
from src.evaluation.evaluator import DatasetEvaluator
from src.evaluation.metrics import is_accepted
from src.execution.code_runner import CodeRunner
from src.llm.openai_client import OpenAIGeneratorClient
from src.llm.prompt_builder import PromptBuilder
from src.storage.csv_store import save_dataset_csv
from src.storage.json_store import save_json
from src.storage.log_store import JsonlLogStore
from src.storage.path_manager import PathManager


class AttemptRunner:
    """
    Run a single attempt for one target cell and one target dataset slot.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        model_config: ModelConfig,
        path_manager: PathManager,
        prompt_builder: PromptBuilder,
        llm_client: OpenAIGeneratorClient,
        code_runner: CodeRunner,
        evaluator: DatasetEvaluator,
        log_store: JsonlLogStore,
    ) -> None:
        self.experiment_config = experiment_config
        self.model_config = model_config
        self.path_manager = path_manager
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.code_runner = code_runner
        self.evaluator = evaluator
        self.log_store = log_store

    def run_initial_attempt(
        self,
        *,
        run_id: str,
        cell: Cell,
        target_dataset_index: int,
        attempt_index: int,
        is_last_x_bin: bool,
        is_last_y_bin: bool,
    ) -> AttemptOutcome:
        """
        Run one initial attempt for the given target cell.
        """
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_initial_prompt(cell)

        llm_response, llm_runtime_seconds = self.llm_client.generate_initial(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return self._finalize_attempt(
            run_id=run_id,
            cell=cell,
            target_dataset_index=target_dataset_index,
            attempt_index=attempt_index,
            llm_response=llm_response,
            llm_runtime_seconds=llm_runtime_seconds,
            is_last_x_bin=is_last_x_bin,
            is_last_y_bin=is_last_y_bin,
        )

    def run_repair_attempt(
        self,
        *,
        run_id: str,
        cell: Cell,
        target_dataset_index: int,
        attempt_index: int,
        achieved_x: float,
        achieved_y: float,
        previous_response_id: str,
        is_last_x_bin: bool,
        is_last_y_bin: bool,
    ) -> AttemptOutcome:
        """
        Run one repair attempt after a previous miss.
        """
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_repair_prompt(
            cell=cell,
            achieved_x=achieved_x,
            achieved_y=achieved_y,
        )

        llm_response, llm_runtime_seconds = self.llm_client.generate_repair(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            previous_response_id=previous_response_id,
        )

        return self._finalize_attempt(
            run_id=run_id,
            cell=cell,
            target_dataset_index=target_dataset_index,
            attempt_index=attempt_index,
            llm_response=llm_response,
            llm_runtime_seconds=llm_runtime_seconds,
            is_last_x_bin=is_last_x_bin,
            is_last_y_bin=is_last_y_bin,
        )

    def _finalize_attempt(
        self,
        *,
        run_id: str,
        cell: Cell,
        target_dataset_index: int,
        attempt_index: int,
        llm_response,
        llm_runtime_seconds: float,
        is_last_x_bin: bool,
        is_last_y_bin: bool,
    ) -> AttemptOutcome:
        """
        Shared logic for executing, evaluating, saving, and logging one attempt.
        """
        attempt_id = make_attempt_id(
            cell.cell_id,
            target_dataset_index=target_dataset_index,
            attempt_index=attempt_index,
        )
        dataset_id = make_dataset_id(cell.cell_id, accepted_index=target_dataset_index)

        execution_result = self.code_runner.run(
            python_code=llm_response.python_code,
            seed=self.experiment_config.generation.dataset_seed_start,
        )

        evaluation_result = None
        accepted = False
        code_path = None
        csv_path = None

        if execution_result.success:
            evaluation_result = self.evaluator.evaluate(execution_result.X, execution_result.y)

            accepted = is_accepted(
                cell,
                evaluation_result,
                is_last_x_bin=is_last_x_bin,
                is_last_y_bin=is_last_y_bin,
            )

            if accepted:
                artifact_dir = self.path_manager.get_dataset_dir(cell.cell_id, dataset_id)
            else:
                artifact_dir = self.path_manager.get_attempt_dir(cell.cell_id, attempt_id)

            code_path = artifact_dir / make_generator_filename(attempt_id)
            metadata_name = (
                make_metadata_filename(dataset_id) if accepted else f"{attempt_id}.json"
            )
            metadata_path = artifact_dir / metadata_name

            code_path.write_text(llm_response.python_code, encoding="utf-8")

            csv_filename = (
                make_csv_filename(dataset_id) if accepted else f"{attempt_id}.csv"
            )
            csv_path = artifact_dir / csv_filename
            save_dataset_csv(execution_result.X, execution_result.y, csv_path)

            metadata = {
                "run_id": run_id,
                "cell_id": cell.cell_id,
                "target_dataset_index": target_dataset_index,
                "attempt_index": attempt_index,
                "attempt_id": attempt_id,
                "dataset_id": dataset_id,
                "accepted": accepted,
                "response_id": llm_response.response_id,
                "mechanism_brief": llm_response.mechanism_brief,
                "expected_x_behavior": llm_response.expected_x_behavior,
                "expected_y_behavior": llm_response.expected_y_behavior,
                "x_score": evaluation_result.x_score,
                "y_score": evaluation_result.y_score,
                "n_rows": evaluation_result.n_rows,
                "n_cols": evaluation_result.n_cols,
                "llm_runtime_seconds": llm_runtime_seconds,
                "code_runtime_seconds": execution_result.runtime_seconds,
                "evaluation_runtime_seconds": evaluation_result.runtime_seconds,
                "total_runtime_seconds": (
                    llm_runtime_seconds
                    + execution_result.runtime_seconds
                    + evaluation_result.runtime_seconds
                ),
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
                "total_tokens": llm_response.total_tokens,
            }
            save_json(metadata, metadata_path)

        total_runtime_seconds = llm_runtime_seconds + execution_result.runtime_seconds
        if evaluation_result is not None:
            total_runtime_seconds += evaluation_result.runtime_seconds

        attempt_record = AttemptRecord(
                run_id=run_id,
                cell_id=cell.cell_id,
                target_dataset_index=target_dataset_index,
                attempt_index=attempt_index,
                accepted=accepted,
                exhausted=False,
                response_id=llm_response.response_id,
                x_score=None if evaluation_result is None else evaluation_result.x_score,
                y_score=None if evaluation_result is None else evaluation_result.y_score,
                error_type=execution_result.error_type,
                error_message=execution_result.error_message,
                generator_code_path=None if code_path is None else str(code_path),
                dataset_csv_path=None if csv_path is None else str(csv_path),
                llm_runtime_seconds=llm_runtime_seconds,
                code_runtime_seconds=execution_result.runtime_seconds,
                evaluation_runtime_seconds=(
                    None if evaluation_result is None else evaluation_result.runtime_seconds
                ),
                total_runtime_seconds=total_runtime_seconds,
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
                total_tokens=llm_response.total_tokens,
            )

        self.log_store.append(attempt_record, self.path_manager.get_attempts_log_path())

        return AttemptOutcome(
            attempt_record=attempt_record,
            accepted=accepted,
            llm_response=llm_response,
            execution_result=execution_result,
            evaluation_result=evaluation_result,
        )