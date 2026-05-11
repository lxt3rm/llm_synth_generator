# LLM Synthetic Dataset Generator

Code for the **propose–execute–evaluate–repair** loop described in
Zhu & Ler, *LLM-Driven Performance-Space Augmentation for Meta-learning-Based
Algorithm Selection* (KDD '26 Undergraduate Consortium submission).

This repository reproduces the synthetic dataset pool used by the paper, the
**5x5 grid (248 datasets)**, the **7x7 grid (482 datasets)**, and their
**combined 730-dataset pool**, that are described in Section 3.2 of the
paper (Proposed LLM Synthetic Dataset Generator) and consumed by Section 4
(Experimental Setup).

An LLM is asked to return an executable Python procedure
`generate(seed) -> (X, y)`. The procedure runs in a sandboxed subprocess
under a fixed evaluation harness that scores each candidate against the
target cell of the 2-D performance space
`φ(D) = (R²_KNN, R²_LR) ∈ [0, 1]²`. If the candidate falls outside the cell,
a repair prompt is sent on the same conversational thread until the
per-cell attempt budget is exhausted.

## Pipeline position

This repo is **stage 1 of 3**. Its accepted datasets are consumed by
[meta_learner_experiments/create_meta_dataset/](https://github.com/lxt3rm/meta_learner_experiments/tree/main/create_meta_dataset),
which computes meta-features and meta-labels for each synthetic dataset and
produces the `syn_meta_dataset_{248,482}.csv` files. Those files are then
consumed by the top-level scripts of
[meta_learner_experiments](https://github.com/lxt3rm/meta_learner_experiments),
which run the no-augmentation / uniform / margin evaluation, the H1/H2
paired *t*-tests, and the learning-curve figures.

## Environment

- Python ≥ 3.11.
- OpenAI API key. The OpenAI client errors if `OPENAI_API_KEY` is unset.
- CPU only. The evaluation harness uses scikit-learn pipelines (KNN, linear
  regression) and never requires a GPU.

```bash
cd llm_synth_generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Repository structure

| Path | Purpose | Paper anchor |
| --- | --- | --- |
| `src/main.py` | Entry point. Loads the three YAML configs, builds the grid, runs all selected cells, writes the run summary. | Section 3.2 |
| `src/orchestration/` | `attempt_runner.py`, `cell_runner.py`, `experiment_runner.py`. The propose–execute–evaluate–repair retry loop with deterministic per-cell / per-target execution seeds. | Section 3.2 |
| `src/llm/` | `openai_client.py` (Responses API; `previous_response_id` continues the repair thread), `prompt_builder.py`, `response_parser.py`, `schemas.py` (strict JSON schema for the response). | Section 3.2 |
| `src/execution/` | `code_runner.py` runs the LLM-emitted code in a `spawn` subprocess with a restricted `__import__` (`numpy`, `numpy.linalg` only), a curated builtins set, no file I/O, and a wall-clock timeout. `validator.py` enforces the `(X, y)` shape contract. | Section 3.2 |
| `src/evaluation/` | `harness.py` defines the fixed acceptance harness (KNN + LinearRegression pipelines, both with `MinMaxScaler`, scored by mean R² over a deterministic 5-fold totally-stratified CV). `tscv.py` is the totally-stratified splitter. `metrics.py` performs the cell-membership test. `evaluator.py` exposes `evaluate(X, y)` to generated code. | Section 3.2 |
| `src/core/` | `grid.py` builds the rectangular grid, `cell_selection.py` filters it, `config.py` loads and validates the YAML, `types.py` defines the dataclasses, `ids.py` derives stable cell / dataset / attempt IDs, `constants.py` holds shared constants. | Section 3.2 |
| `src/storage/` | `csv_store.py`, `json_store.py`, `log_store.py` (JSONL), `path_manager.py`. | Section 3.2 |
| `src/analysis/` | `load_logs.py`, `summarise_attempts.py`. Post-run aggregation of `attempts.jsonl`. | Code-only. |
| `prompts/` | `system_prompt.txt`, `initial_prompt.txt`, `repair_prompt.txt`. | Section 3.2 (referenced in the paper as the prompt templates). |
| `configs/` | `grid.yaml` (axis ranges, `bins_per_axis`), `experiment.yaml` (run name, paths, generation budget, seed, cell selection, execution timeout), `model.yaml` (model name, temperature, max output tokens). | Section 3.2 (`w = 10`, `b = 84`); Section 4 (5x5 / 7x7 grids). |

## Reproducing the synthetic dataset pool

Run from inside `llm_synth_generator/`.

### 5x5 grid (≈248 datasets)

```bash
# In configs/grid.yaml: set bins_per_axis: 5
# In configs/experiment.yaml: set run_name: "grid_5x5", cell_selection.mode: "all"
python -m src.main
```

### 7x7 grid (≈482 datasets)

```bash
# In configs/grid.yaml: set bins_per_axis: 7
# In configs/experiment.yaml: set run_name: "grid_7x7", cell_selection.mode: "all"
python -m src.main
```

### Combined pool (730 datasets)

The combined pool is the union of `data/accepted/` from both runs. Cells in
unreachable regions of the performance space (e.g. simultaneously high
`R²_KNN` and very low `R²_LR`) may exhaust their per-cell attempt budget
before collecting all `w = 10` witnesses, so the exact counts (248 / 482)
are run-dependent.

To pass the pool to the downstream meta-feature/meta-label stage, point
[create_meta_dataset](https://github.com/lxt3rm/meta_learner_experiments/tree/main/create_meta_dataset)
at the `data/accepted/` folder of each grid in turn.

### Outputs

| Path | Contents |
| --- | --- |
| `data/accepted/<cell_id>/<dataset_id>/` | Accepted dataset CSV (columns `x1, …, xd, y`), the generator code (`*.py`), and a metadata JSON. |
| `data/rejected/<cell_id>/<attempt_id>/` | Rejected attempts (CSV + code + metadata). |
| `logs/attempts.jsonl` | One JSON line per attempt (initial and repair). |
| `logs/run_summary.json` | Per-cell and overall accepted / attempted counts and hit rate. |
| `logs/attempt_analysis.json` | Post-run aggregates: mean attempts per accepted dataset, runtime, token totals. |

The accepted CSV schema (`x1, ..., xd, y`) is consumed verbatim by
`create_meta_dataset`, which assumes the last column is the regression
target.

## Configuration

| File | Key | Default | Effect |
| --- | --- | --- | --- |
| `configs/grid.yaml` | `bins_per_axis` | `7` | Bins on each axis of the 2-D performance grid. |
| `configs/grid.yaml` | `x.min`, `x.max` | `0.0`, `1.0` | Range of the x-axis (KNN R²). |
| `configs/grid.yaml` | `y.min`, `y.max` | `0.0`, `1.0` | Range of the y-axis (Linear Regression R²). |
| `configs/experiment.yaml` | `run_name` | `"pilot_7x7_v1"` | Used in the run ID and printed in the summary. |
| `configs/experiment.yaml` | `paths.accepted_dir` | `data/accepted` | Where accepted datasets and their generators are written. |
| `configs/experiment.yaml` | `paths.rejected_dir` | `data/rejected` | Where rejected attempts are written. |
| `configs/experiment.yaml` | `paths.summaries_dir` | `data/summaries` | Created at startup. Reserved for downstream summaries. |
| `configs/experiment.yaml` | `paths.logs_dir` | `logs` | Where `attempts.jsonl`, `run_summary.json`, and `attempt_analysis.json` land. |
| `configs/experiment.yaml` | `generation.accepted_per_cell` | `10` | Target witnesses per cell (paper's `w = 10`). |
| `configs/experiment.yaml` | `generation.max_total_attempts_per_cell` | `84` | Hard attempt budget per cell (paper's `b = 84`). |
| `configs/experiment.yaml` | `generation.max_retries_per_dataset` | `7` | Maximum repair attempts for one target dataset slot. |
| `configs/experiment.yaml` | `generation.dataset_seed_start` | `11` | Base seed for deterministic execution-seed derivation. Retries for one slot reuse the same seed. |
| `configs/experiment.yaml` | `execution.timeout_seconds` | `300` | Wall-clock cap for one `generate(seed)` call. |
| `configs/experiment.yaml` | `cell_selection.mode` | `"all"` | One of `all`, `include_ids`, or `row_col_ranges`. |
| `configs/experiment.yaml` | `cell_selection.cell_ids` | (unset) | Required when `mode == "include_ids"`. Cell IDs follow the format `cell_<row:02d>_<col:02d>`. |
| `configs/experiment.yaml` | `cell_selection.row_indices`, `cell_selection.col_indices` | (unset) | Required when `mode == "row_col_ranges"`. |
| `configs/model.yaml` | `model.name` | `"gpt-5.4"` | OpenAI model used by the Responses API. Matches Section 3.2 of the paper verbatim. |
| `configs/model.yaml` | `model.temperature` | `1.0` | Sampling temperature passed at request build time. |
| `configs/model.yaml` | `model.max_output_tokens` | `4000` | Per-request output token cap. |

## Hardware

The provided experiments ran on a machine with **32 GiB RAM and 8 vCPUs**,
CPU only. The acceptance harness is lightweight (a single 5-fold CV over
KNN and linear regression per candidate), so wall-clock is dominated by the
OpenAI API round-trip. No GPU is required.
