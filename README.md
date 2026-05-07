# LLM-Driven Performance-Space Augmentation for Meta-learning-Based Algorithm Selection

Official code release for the **LLM Synthetic Dataset Generator** described in Zhu & Ler, *LLM-Driven Performance-Space Augmentation for Meta-learning-Based Algorithm Selection* (preprint).

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](#)
<!-- TODO: replace the link target above with the arXiv URL once posted. -->
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-3776AB)](https://www.python.org/)
<!-- TODO: pin once a pyproject.toml is added; floor inferred from the requirements.txt pins (numpy 2.4.x requires Python >= 3.11). -->
[![OpenAI SDK](https://img.shields.io/badge/openai-2.28.0-412991)](https://github.com/openai/openai-python)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E)](https://scikit-learn.org/)

<!-- TODO: add a pipeline diagram here showing propose -> execute -> evaluate -> repair around the 2-D performance grid. -->

## TL;DR

Meta-learning for algorithm selection is bottlenecked by the small number of curated regression datasets, which leaves the meta-dataset sparse and the meta-learner under-generalising. This repository implements the **propose–execute–evaluate–repair** loop that produces synthetic regression datasets steered toward target cells of a two-dimensional **performance space** `φ(D) = (R²_KNN, R²_LR) ∈ [0,1]²`. An LLM is asked to return an executable Python procedure `generate(seed) -> (X, y)`; the procedure runs in a sandboxed subprocess with a restricted set of builtins and `numpy`-only imports; the resulting dataset is scored under a fixed harness; and a repair prompt continues the same conversational thread until the dataset lands inside the target cell or the per-cell attempt budget is exhausted. The synthetic pool produced here is the input to the downstream meta-learning evaluation reported in the paper. **That evaluation pipeline (regression and multi-label meta-learners, uniform vs. margin-based augmentation, learning curves, Monte Carlo, paired t-tests) is not part of this repository.**

## Installation

Requires Python >= 3.11.

```bash
cd llm_synth_generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # required; the client errors out if unset
```

## Quick start

The only entry point is `src.main`, which loads its three YAML configs by relative path. Run it from inside `llm_synth_generator/`:

```bash
cd llm_synth_generator
export OPENAI_API_KEY=sk-...

# Optional: trim the run to one or two cells before the first end-to-end test.
# Edit configs/experiment.yaml and replace cell_selection with:
#   cell_selection:
#     mode: "include_ids"
#     cell_ids: ["cell_03_03"]

python -m src.main
```

There is no separate "small sample" script; the budget is controlled by `cell_selection`, `generation.accepted_per_cell`, and `generation.max_total_attempts_per_cell` in `configs/experiment.yaml`.

Outputs:

| Path | Contents |
| --- | --- |
| `data/accepted/<cell_id>/<dataset_id>/` | Accepted dataset CSV, generator code (`*.py`), and metadata JSON. |
| `data/rejected/<cell_id>/<attempt_id>/` | Rejected attempts (CSV + code + metadata). |
| `logs/attempts.jsonl` | One JSON line per attempt (initial and repair). |
| `logs/run_summary.json` | Per-cell and overall accepted / attempted counts and hit rate. |
| `logs/attempt_analysis.json` | Post-run aggregates: mean attempts per accepted dataset, runtime, token totals. |

## Repository structure

| Path | Purpose | Paper anchor |
| --- | --- | --- |
| `src/main.py` | Entry point; wires configs, builds the grid, runs all selected cells, summarises. | Section 3.2 (orchestration) |
| `src/orchestration/` | `attempt_runner.py`, `cell_runner.py`, `experiment_runner.py` — the propose–execute–evaluate–repair retry loop, with deterministic per-cell / per-target execution seeds. | Section 3.2 ("The Repair Loop") |
| `src/llm/` | `openai_client.py` (OpenAI Responses API with `previous_response_id` for repair-thread continuity), `prompt_builder.py`, `response_parser.py`, `schemas.py` (strict JSON schema for the response). | Section 3.2 (prompting) |
| `src/execution/` | `code_runner.py` runs the LLM-emitted code in a `spawn` subprocess with a restricted `__import__` (only `numpy` / `numpy.linalg`), a curated builtins set, no file I/O, and a wall-clock timeout. `validator.py` enforces the `(X, y)` shape contract. | Section 3.2 ("Execution and Validation") |
| `src/evaluation/` | `harness.py` defines the fixed acceptance harness (KNN and LinearRegression pipelines, both with `MinMaxScaler`, scored by mean R² over a deterministic 5-fold totally-stratified CV). `tscv.py` implements the totally-stratified splitter. `metrics.py` performs the cell-membership acceptance test. `evaluator.py` exposes `evaluate(X, y)` to generated code. | Section 3.2 ("The Repair Loop"); Section 3.3 (TSCV reference) |
| `src/core/` | `grid.py` builds the rectangular grid; `cell_selection.py` filters it; `config.py` loads and validates the YAML; `types.py` defines the dataclasses; `ids.py` derives stable cell / dataset / attempt IDs; `constants.py` holds shared constants. | Section 3.2 (grid discretisation) |
| `src/storage/` | `csv_store.py`, `json_store.py`, `log_store.py` (JSONL), `path_manager.py`. | Section 3.2 (artefact persistence) |
| `src/analysis/` | `load_logs.py`, `summarise_attempts.py` — post-run aggregation of `attempts.jsonl`. | Code-only; not described in the paper. |
| `prompts/` | `system_prompt.txt`, `initial_prompt.txt`, `repair_prompt.txt` — verbatim prompt templates. | Appendix A.2 ("LLM Prompts") |
| `configs/` | `grid.yaml` (axis ranges, `bins_per_axis`), `experiment.yaml` (run name, paths, generation budget, seed, cell selection, execution timeout), `model.yaml` (model name, temperature, max output tokens). | Section 3.2 (`w = 10`, `b = 84`); Section 4 (5x5 / 7x7 grids) |

## Reproducing paper results

This repository reproduces only the **synthetic dataset pool generation** stage of the paper. The 5x5 grid (≈248 datasets) and 7x7 grid (≈482 datasets) that combine into the 730-dataset pool used in the paper's evaluation are regenerated here. The downstream meta-learning evaluation that produces the ablation table, the learning curves, the Monte Carlo selection-frequency analysis, and the H1 / H2 paired t-tests lives in a separate codebase that will be released alongside the paper.

<!-- TODO: replace with the URL to the companion evaluation repository once published. -->
**Companion evaluation repository:** TBD.

### 5x5 grid (~248 datasets)

```bash
# 1. In configs/grid.yaml, set:
#      bins_per_axis: 5
# 2. In configs/experiment.yaml, set:
#      run_name: "grid_5x5"
#      cell_selection:
#        mode: "all"
python -m src.main
```

Acceptance budget per cell: `accepted_per_cell: 10` and `max_total_attempts_per_cell: 84` (matching the paper's `w = 10`, `b = 84`).

### 7x7 grid (~482 datasets)

```bash
# 1. In configs/grid.yaml, set:
#      bins_per_axis: 7      # this is the shipped default
# 2. In configs/experiment.yaml, set:
#      run_name: "grid_7x7"
#      cell_selection:
#        mode: "all"
python -m src.main
```

### Combined pool

Concatenating the contents of `data/accepted/` from both runs reproduces the 730-dataset pool used in the paper. The exact counts (248 / 482) are run-dependent: cells in unreachable regions of the performance space (e.g. `R²_KNN = 1` simultaneously with `R²_LR <= 0`) may exhaust their per-cell attempt budget before collecting all `w = 10` witnesses.

## Configuration

| File | Key | Default | Effect |
| --- | --- | --- | --- |
| `configs/grid.yaml` | `bins_per_axis` | `7` | Number of bins on each axis of the 2-D performance grid. |
| `configs/grid.yaml` | `x.min`, `x.max` | `0.0`, `1.0` | Range of the x-axis (KNN R²). |
| `configs/grid.yaml` | `y.min`, `y.max` | `0.0`, `1.0` | Range of the y-axis (Linear Regression R²). |
| `configs/experiment.yaml` | `run_name` | `"pilot_7x7_v1"` | Used in the run ID and printed in the summary. |
| `configs/experiment.yaml` | `paths.accepted_dir` | `data/accepted` | Where accepted datasets and their generators are written. |
| `configs/experiment.yaml` | `paths.rejected_dir` | `data/rejected` | Where rejected attempts are written. |
| `configs/experiment.yaml` | `paths.summaries_dir` | `data/summaries` | Created at startup; reserved for downstream summaries. |
| `configs/experiment.yaml` | `paths.logs_dir` | `logs` | Where `attempts.jsonl`, `run_summary.json`, and `attempt_analysis.json` land. |
| `configs/experiment.yaml` | `generation.accepted_per_cell` | `10` | Target witnesses per cell (paper's `w`). |
| `configs/experiment.yaml` | `generation.max_total_attempts_per_cell` | `84` | Hard attempt budget per cell (paper's `b`). |
| `configs/experiment.yaml` | `generation.max_retries_per_dataset` | `7` | Maximum repair attempts for one target dataset slot. |
| `configs/experiment.yaml` | `generation.dataset_seed_start` | `11` | Base seed for deterministic execution-seed derivation; retries for one slot reuse the same seed. |
| `configs/experiment.yaml` | `execution.timeout_seconds` | `300` | Wall-clock cap for one `generate(seed)` call. |
| `configs/experiment.yaml` | `cell_selection.mode` | `"all"` | One of `all`, `include_ids`, or `row_col_ranges`. |
| `configs/experiment.yaml` | `cell_selection.cell_ids` | (unset) | Required when `mode == "include_ids"`. Cell IDs follow the format `cell_<row:02d>_<col:02d>`. |
| `configs/experiment.yaml` | `cell_selection.row_indices`, `cell_selection.col_indices` | (unset) | Required when `mode == "row_col_ranges"`. |
| `configs/model.yaml` | `model.name` | `"gpt-5.4"` | OpenAI model used by the Responses API. <!-- TODO: verify the intended OpenAI model name; the paper does not specify one. --> |
| `configs/model.yaml` | `model.temperature` | `1.0` | Sampling temperature passed at request build time. |
| `configs/model.yaml` | `model.max_output_tokens` | `4000` | Per-request output token cap. |

