"""
Utilities for saving JSON artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_serializable(obj: Any) -> Any:
    """
    Convert common Python objects into JSON-serializable forms.
    """
    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Save an object as a JSON file.

    Dataclasses are converted automatically using asdict().
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, default=_to_serializable, indent=indent)