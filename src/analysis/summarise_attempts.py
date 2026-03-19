"""
Summarise attempt-level logs into report-ready statistics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.analysis.load_logs import load_jsonl
from src.storage.json_store import save_json


def _safe_mean(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None if the list is empty."""
    if not values:
        return None
    return sum(values) / len(values)


def summarise_attempt_log(attempt_log_path: str | Path) -> dict:
    """
    Compute summary statistics from the attempts.jsonl log.
    """
    records = load_jsonl(attempt_log_path)

    total_attempts = len(records)
    accepted_records = [r for r in records if r.get("accepted") is True]
    total_accepted = len(accepted_records)

    overall_hit_rate = total_accepted / total_attempts if total_attempts > 0 else 0.0

    llm_times = [
        float(r["llm_runtime_seconds"])
        for r in records
        if r.get("llm_runtime_seconds") is not None
    ]
    code_times = [
        float(r["code_runtime_seconds"])
        for r in records
        if r.get("code_runtime_seconds") is not None
    ]
    eval_times = [
        float(r["evaluation_runtime_seconds"])
        for r in records
        if r.get("evaluation_runtime_seconds") is not None
    ]
    total_times = [
        float(r["total_runtime_seconds"])
        for r in records
        if r.get("total_runtime_seconds") is not None
    ]

    input_tokens = [
        int(r["input_tokens"])
        for r in records
        if r.get("input_tokens") is not None
    ]
    output_tokens = [
        int(r["output_tokens"])
        for r in records
        if r.get("output_tokens") is not None
    ]
    total_tokens = [
        int(r["total_tokens"])
        for r in records
        if r.get("total_tokens") is not None
    ]

    attempts_per_accepted_dataset: dict[tuple[str, int], int] = defaultdict(int)
    accepted_keys: set[tuple[str, int]] = set()

    for r in records:
        key = (r["cell_id"], int(r["target_dataset_index"]))
        attempts_per_accepted_dataset[key] += 1
        if r.get("accepted") is True:
            accepted_keys.add(key)

    accepted_attempt_counts = [
        attempts_per_accepted_dataset[key]
        for key in accepted_keys
    ]

    per_cell = defaultdict(lambda: {
        "attempts": 0,
        "accepted": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_runtime_seconds": 0.0,
    })

    for r in records:
        cell_id = r["cell_id"]
        per_cell[cell_id]["attempts"] += 1
        per_cell[cell_id]["accepted"] += int(bool(r.get("accepted")))
        per_cell[cell_id]["input_tokens"] += int(r["input_tokens"] or 0)
        per_cell[cell_id]["output_tokens"] += int(r["output_tokens"] or 0)
        per_cell[cell_id]["total_tokens"] += int(r["total_tokens"] or 0)
        per_cell[cell_id]["total_runtime_seconds"] += float(r["total_runtime_seconds"] or 0.0)

    per_cell_summary = {}
    for cell_id, stats in per_cell.items():
        attempts = stats["attempts"]
        accepted = stats["accepted"]
        per_cell_summary[cell_id] = {
            "attempts": attempts,
            "accepted": accepted,
            "hit_rate": accepted / attempts if attempts > 0 else 0.0,
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "total_tokens": stats["total_tokens"],
            "total_runtime_seconds": stats["total_runtime_seconds"],
            "mean_runtime_per_attempt": (
                stats["total_runtime_seconds"] / attempts if attempts > 0 else None
            ),
        }

    summary = {
        "total_attempts": total_attempts,
        "total_accepted": total_accepted,
        "overall_hit_rate": overall_hit_rate,
        "mean_attempts_per_accepted_dataset": _safe_mean(accepted_attempt_counts),
        "mean_llm_runtime_seconds": _safe_mean(llm_times),
        "mean_code_runtime_seconds": _safe_mean(code_times),
        "mean_evaluation_runtime_seconds": _safe_mean(eval_times),
        "mean_total_runtime_seconds": _safe_mean(total_times),
        "sum_input_tokens": sum(input_tokens),
        "sum_output_tokens": sum(output_tokens),
        "sum_total_tokens": sum(total_tokens),
        "mean_input_tokens_per_attempt": _safe_mean(input_tokens),
        "mean_output_tokens_per_attempt": _safe_mean(output_tokens),
        "mean_total_tokens_per_attempt": _safe_mean(total_tokens),
        "per_cell": dict(sorted(per_cell_summary.items())),
    }

    return summary


def save_attempt_summary(
    attempt_log_path: str | Path,
    output_path: str | Path,
) -> dict:
    """
    Summarise the attempt log and save the result as JSON.
    """
    summary = summarise_attempt_log(attempt_log_path)
    save_json(summary, output_path)
    return summary