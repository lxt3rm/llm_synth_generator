"""
Utilities for loading JSONL experiment logs.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Load a JSONL file into a list of dictionaries.

    Empty lines are ignored.
    """
    log_path = Path(path)

    if not log_path.exists():
        raise FileNotFoundError(f"JSONL log file not found: {log_path}")

    records: list[dict] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records