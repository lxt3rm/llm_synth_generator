# LLM Synthetic Dataset Generator

This repository runs an experiment in which an LLM proposes Python dataset generators for synthetic regression problems, executes the generated code, evaluates each dataset under a fixed model-comparison harness, and keeps only the generations that land in a target region of a 2D performance space. In the current codebase, the two axes are mean cross-validated `R^2` for a KNN regressor and a linear regressor, and the experiment searches for datasets that populate cells of a rectangular grid over that space.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_KEY_HERE
python -m src.main
```

Outputs are written under `data/` and `logs/`.

## Research Goal

The repository is set up to test whether an LLM can iteratively generate synthetic regression datasets whose downstream behavior matches a requested target cell in a discretized performance grid.

Concretely, the current experiment:

- Defines a 2D space where:
  - `x_score` = mean cross-validated `R^2` of a KNN regressor
  - `y_score` = mean cross-validated `R^2` of a linear regressor
- Partitions that space into a `5 x 5` grid by default
- Asks the LLM to write `generate(seed) -> (X, y)` code that should place the dataset inside a requested cell
- Retries with a repair prompt when an attempt misses the target
- Saves accepted and rejected generations for later inspection
- Produces run-level and attempt-level summaries

## Core Idea

The experiment loop is implemented in code rather than notebooks. The intended flow is:

1. Pick a target cell in the `(x_score, y_score)` grid.
2. Build an initial prompt describing the target region.
3. Ask the OpenAI Responses API for structured output containing:
   - a short mechanism description
   - Python code for `generate(seed: int)`
   - expected behavior along both evaluation axes
4. Execute the returned code in a restricted subprocess sandbox.
5. Validate that the generator returned numeric `X` and `y` with the expected shapes.
6. Evaluate the dataset using the fixed harness in `src/evaluation/`.
7. Accept the attempt if the measured `(x_score, y_score)` falls inside the requested cell.
8. Otherwise, send a repair prompt that includes the achieved scores and try again until either:
   - the required number of accepted datasets is collected for that cell, or
   - the retry / attempt budget is exhausted.

## Repository Structure

```text
configs/
  experiment.yaml        Main run settings: output paths, retry budgets, base seed, cell selection
  grid.yaml              Performance-space bounds and number of bins per axis
  model.yaml             OpenAI model name and token budget

prompts/
  system_prompt.txt      Base instructions for the generator model
  initial_prompt.txt     Prompt template for the first attempt in a cell
  repair_prompt.txt      Prompt template for retries after a miss

src/
  main.py                Entry point for the end-to-end experiment
  core/                  Typed configs, IDs, grid construction, cell selection
  llm/                   OpenAI client, response parsing, output schema, prompt builder
  execution/             Generated-code execution sandbox and dataset validation
  evaluation/            Fixed scoring harness and acceptance logic
  orchestration/         Attempt-, cell-, and experiment-level retry loops
  storage/               CSV, JSON, JSONL, and output-path helpers
  analysis/              Attempt-log summarization utilities

data/
  accepted/              Accepted datasets and their metadata/code artifacts
  rejected/              Rejected attempts and their artifacts
  summaries/             Created by config/path setup; currently unused by `src.main`

logs/
  attempts.jsonl         One JSON object per generation attempt
  run_summary.json       Per-run aggregate summary across selected cells
  attempt_analysis.json  Post-run summary derived from `attempts.jsonl`

notebooks/
  (currently empty)
```

## Python Version

The repository does not pin an interpreter version in `pyproject.toml`, `setup.py`, or a `.python-version` file. Based on the source code, Python `3.10+` is required because the code uses modern union syntax such as `str | None`.

If you want the safest choice, use a recent Python 3.10 or 3.11 environment.

## Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies from `requirements.txt` include:

- `openai`
- `numpy`
- `pandas`
- `scikit-learn`
- `PyYAML`

## Environment Variables and External Services

This project requires access to the OpenAI API.

Set:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
```

Important notes:

- `OPENAI_API_KEY` is mandatory. `src/llm/openai_client.py` raises an error if it is missing.
- An empty `.env` file exists in the repository root, but the code does not load `.env` automatically. You must export the variable in your shell or load it through your own environment-management tooling.
- The current code uses the OpenAI Responses API via the `openai` Python SDK.

## Configuration

The main run is controlled through three YAML files.

### `configs/experiment.yaml`

Current contents:

```yaml
run_name: "pilot_5x5_v1"

paths:
  accepted_dir: "data/accepted"
  rejected_dir: "data/rejected"
  summaries_dir: "data/summaries"
  logs_dir: "logs"

generation:
  accepted_per_cell: 10
  max_retries_per_dataset: 7
  max_total_attempts_per_cell: 84
  dataset_seed_start: 11

execution:
  timeout_seconds: 300

cell_selection:
  mode: "all"
```

Meaning of the main fields:

- `run_name`: prefix used when creating `run_id`
- `accepted_per_cell`: how many accepted datasets to collect per selected cell
- `max_retries_per_dataset`: maximum attempts for one target dataset slot before moving on
- `max_total_attempts_per_cell`: total attempt budget for an entire cell
- `dataset_seed_start`: base seed used to derive per-cell, per-target execution seeds
- `timeout_seconds`: wall-clock timeout for one generated-code execution
- `cell_selection.mode`: which cells to run

Supported `cell_selection` modes from `src/core/cell_selection.py`:

- `all`
- `include_ids`
- `row_col_ranges`

Example restricted run:

```yaml
cell_selection:
  mode: "include_ids"
  cell_ids:
    - "cell_00_00"
    - "cell_00_01"
```

Or:

```yaml
cell_selection:
  mode: "row_col_ranges"
  row_indices: [0, 1]
  col_indices: [0, 1, 2]
```

### `configs/grid.yaml`

```yaml
x:
  min: 0.0
  max: 1.0

y:
  min: 0.0
  max: 1.0

bins_per_axis: 5
```

This defines a rectangular `5 x 5` grid over the score space `[0, 1] x [0, 1]`.

### `configs/model.yaml`

```yaml
model:
  name: "gpt-5.4"
  temperature: 1.0
  max_output_tokens: 4000
```

In the current code:

- `name` is used when calling the OpenAI API
- `max_output_tokens` is used
- `temperature` exists in the config dataclass but is not currently passed into `client.responses.create(...)`

## Input Contract for Generated Code

The LLM is asked to return code that defines:

```python
generate(seed: int) -> tuple[np.ndarray, np.ndarray]
```

The generated function must:

- be deterministic given the seed
- return `(X, y)`
- return a 2D numeric `X`
- return a 1D numeric `y`
- avoid file I/O and external dependencies

Inside the execution sandbox, generated code has access to:

- `np` for NumPy
- `evaluate(X, y)` to score candidate datasets against the fixed harness

The sandbox allows only restricted imports, effectively limited to NumPy-related modules used by the runtime.

## Evaluation Protocol

The evaluation logic lives in `src/evaluation/`.

### Models

From `src/evaluation/harness.py`:

- `x_score`: `MinMaxScaler` + `KNeighborsRegressor`
- `y_score`: `MinMaxScaler` + `LinearRegression`

### Cross-validation

The evaluator uses a custom regression splitter in `src/evaluation/tscv.py`.

Current settings in `make_splits(...)`:

- `k = 5` folds
- `repeats = 1`
- `seed = 11`

The splitter sorts examples by `y` and distributes them across folds to create a stratified-like K-fold procedure for regression targets.

### Acceptance Rule

Acceptance is implemented in `src/evaluation/metrics.py`.

- For non-terminal bins, a score must satisfy `[lower, upper)`
- For the last bin on an axis, the upper bound is inclusive: `[lower, upper]`

An attempt is accepted only if both:

- `x_score` falls within the target cell's x-range
- `y_score` falls within the target cell's y-range

## How the Pipeline Works

`python -m src.main` performs the following steps:

1. Load experiment, grid, and model configs.
2. Build the full grid with `src/core/grid.py`.
3. Select the subset of cells requested by `cell_selection`.
4. Create output directories if needed.
5. For each cell:
   - request an initial generation
   - execute the returned code
   - validate `X` and `y`
   - evaluate the dataset
   - save accepted or rejected artifacts
   - log the attempt
   - retry with a repair prompt if necessary
6. Save `logs/run_summary.json`.
7. Re-read `logs/attempts.jsonl` and save `logs/attempt_analysis.json`.

The repair loop preserves `previous_response_id` and passes it back to the OpenAI Responses API, so retries are threaded as continuations of the earlier response.

## How to Run the Project

There is currently one primary entry point.

```bash
python -m src.main
```

There are no CLI flags in the present implementation. To change run behavior, edit the YAML config files in `configs/`.

## Recommended First Run

Because the default config targets all 25 cells and 10 accepted datasets per cell, a full run can be large and expensive. A smaller pilot is easier for first-time validation.

Example:

```yaml
generation:
  accepted_per_cell: 1
  max_retries_per_dataset: 2
  max_total_attempts_per_cell: 4
  dataset_seed_start: 11

cell_selection:
  mode: "include_ids"
  cell_ids:
    - "cell_00_00"
    - "cell_00_01"
```

Then run:

```bash
python -m src.main
```

## Accepted Datasets

Accepted artifacts are stored under:

```text
data/accepted/<cell_id>/<dataset_id>/
```

Typical contents:

- `<dataset_id>.csv`: the generated dataset, with columns `x1`, `x2`, ..., `xd`, `y`
- `<dataset_id>.json`: metadata including scores, dimensions, response ID, token usage, and runtimes
- `<attempt_id>.py`: the generator code associated with the accepted attempt

Example checked-in path:

```text
data/accepted/cell_00_00/cell_00_00__ds_001/
```

## Rejected Attempts

Rejected artifacts are stored under:

```text
data/rejected/<cell_id>/<attempt_id>/
```

Typical contents:

- `<attempt_id>.csv` if execution succeeded but the dataset missed the target cell
- `<attempt_id>.json` with metadata
- `<attempt_id>.py` with the attempted generator code

If generated code fails before producing a valid dataset, the attempt is still logged in `attempts.jsonl`, but code/data artifacts may be absent because `AttemptRunner` only writes them after successful execution and evaluation.

## Logs

The main log files are:

- `logs/attempts.jsonl`: append-only attempt log with one record per generation attempt
- `logs/run_summary.json`: overall totals and per-cell summaries for the run
- `logs/attempt_analysis.json`: aggregate statistics computed from `attempts.jsonl`

Attempt records include:

- `run_id`
- `cell_id`
- `target_dataset_index`
- `attempt_index`
- `execution_seed`
- acceptance status
- `x_score`, `y_score`
- execution error type/message, if any
- paths to saved code and CSV artifacts
- LLM, execution, and evaluation runtimes
- input/output/total token counts

## Analysis and Summaries

`src/analysis/summarise_attempts.py` computes:

- total attempts
- total accepted
- overall hit rate
- mean attempts per accepted dataset
- mean runtimes
- token totals and per-attempt token means
- per-cell summaries

The current entry point writes the output to:

```text
logs/attempt_analysis.json
```

Although `data/summaries/` is created by the path manager configuration, it is not used by the current `src.main` workflow.

## Reproducibility Notes

Several parts of the experiment are deterministic, but the full pipeline is not fully reproducible in the strict sense.

What is deterministic in the current code:

- Grid construction
- Cell selection
- The evaluation splitter seed (`11`)
- Per-attempt execution seeds derived from `dataset_seed_start`
- Execution timeout and acceptance rules

Important caveats:

- The LLM call itself is not guaranteed to be reproducible across time, even with the same prompts.
- The configured `temperature` is not currently passed into the OpenAI API call.
- Execution seeds are derived deterministically from the target cell and target dataset slot.
- Retries for the same target dataset slot intentionally reuse the same execution seed so repair attempts isolate prompt/code changes rather than seed variation.
- Results may change with OpenAI model updates, SDK changes, scikit-learn changes, or prompt edits.
- `run_id` is timestamp-based, so output paths differ from run to run.

## Current Repository State

The repository already contains example outputs from prior runs:

- `logs/run_summary.json`
- `logs/attempt_analysis.json`
- several accepted/rejected artifacts under `data/`

Those checked-in artifacts are useful as examples, but they do not necessarily match the current config exactly. For example, the committed `logs/run_summary.json` describes a run with 3 selected cells and 3 accepted datasets total, while `configs/experiment.yaml` currently requests all cells and `10` accepted datasets per cell.

## Common Pitfalls and Troubleshooting

- `OPENAI_API_KEY is not set in the environment`
  - Export the variable before running. The repository's `.env` file is not auto-loaded.
- `FileNotFoundError` for config or prompt files
  - Run from the repository root so relative paths like `configs/experiment.yaml` and `prompts/system_prompt.txt` resolve correctly.
- Run appears much slower or more expensive than expected
  - Reduce `accepted_per_cell`, `max_retries_per_dataset`, and the number of selected cells for pilot runs.
- Generated code times out
  - The timeout is controlled by `execution.timeout_seconds` in `configs/experiment.yaml`.
- No accepted datasets for a target cell
  - That can happen if the target region is hard to hit or the attempt budget is too small.
- Validation errors from generated code
  - The generator must return numeric `X` and `y` with shapes `(n_rows, n_features)` and `(n_rows,)`.

## Limitations

The current repository is functional but still clearly experimental.

- Only one main entry point is provided.
- There is no command-line interface; configuration is file-based.
- Only OpenAI-backed generation is implemented.
- The current evaluation harness is fixed to KNN vs. linear regression.
- Prompt templates and response parsing are slightly out of sync:
  - the prompts ask for tagged sections like `<mechanism_brief>...</mechanism_brief>`
  - the OpenAI client requests strict JSON-schema output
  - the parser expects JSON text
- `PathManager` defines paths for `accepted.jsonl` and `failures.jsonl`, but the current orchestration code does not write those logs.
- `data/summaries/` and `notebooks/` are present but not actively used by the current entry point.
- The execution sandbox is restricted, but it is still executing model-generated Python and should be treated cautiously.

## Suggested Next Steps

If you plan to continue this line of work, the most obvious improvements suggested by the current codebase are:

- Add a proper CLI for overriding config values without editing YAML manually
- Resolve the prompt-format mismatch so prompts, schema, and parser all describe the same contract
- Log accepted-only and failure-only streams if `accepted.jsonl` / `failures.jsonl` are desired
- Add plotting or notebook utilities for visualizing coverage across the grid
- Add tests around config loading, acceptance boundaries, and generated-code validation
- Record more provenance about model versioning and prompt revisions for reproducibility

## Minimal End-to-End Example

1. Edit `configs/experiment.yaml` for a small pilot run.
2. Export `OPENAI_API_KEY`.
3. Run:

```bash
python -m src.main
```

4. Inspect:

```text
logs/attempts.jsonl
logs/run_summary.json
logs/attempt_analysis.json
data/accepted/
data/rejected/
```

That is the actual end-to-end workflow implemented in this repository today.
