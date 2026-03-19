"""
Utilities for appending JSONL log records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_loggable(obj: Any) -> Any:
    """
    Convert log objects into JSON-serializable forms.
    """
    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj


class JsonlLogStore:
    """
    Simple JSONL logger.

    Each appended record becomes one line of JSON.
    """

    def append(self, record: Any, path: str | Path) -> None:
        """
        Append one record to a JSONL file.
        """
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_to_loggable, indent=2))
            f.write("\n")