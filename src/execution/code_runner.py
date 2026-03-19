"""
Utilities for executing generated Python code and calling generate(seed).

This uses a moderately permissive sandbox and runs generated code in a
separate process so the generate(seed) call can be capped by a timeout.

It allows common Python builtins, NumPy, and evaluate(X, y), while still
blocking arbitrary imports and file/system access.
"""

from __future__ import annotations

import builtins
import multiprocessing as mp
import queue
import time
from types import FunctionType
from typing import Any

import numpy as np

from src.core.types import ExecutionResult
from src.evaluation.evaluator import DatasetEvaluator
from src.execution.validator import validate_dataset


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    """
    Restricted import function for generated code.

    Only allow a small whitelist of safe numerical modules that may be needed
    by NumPy internals or by generated code.
    """
    allowed_modules = {
        "numpy",
        "numpy.linalg",
    }

    root_name = name.split(".")[0]

    if name in allowed_modules or root_name == "numpy":
        return builtins.__import__(name, globals, locals, fromlist, level)

    raise ImportError(f"Import of module '{name}' is not allowed in generated code")


def _build_safe_builtins() -> dict[str, object]:
    """
    Return a moderately permissive set of builtins for generated code.

    Design goals:
    - allow ordinary Python control flow and container usage
    - allow simple numeric/type conversions
    - allow common iteration helpers
    - block file I/O and dangerous dynamic execution
    - allow only restricted imports
    """
    return {
        # Basic constants and types
        "None": None,
        "True": True,
        "False": False,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,

        # Basic utilities
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "print": print,

        # Iteration / functional helpers
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "reversed": reversed,
        "map": map,
        "filter": filter,
        "any": any,
        "all": all,

        # Introspection / type helpers
        "isinstance": isinstance,
        "issubclass": issubclass,
        "globals": globals,

        # Exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "ImportError": ImportError,

        # Restricted imports only
        "__import__": _restricted_import,
    }


def _worker_run_generated_code(
    python_code: str,
    seed: int,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process entry point.

    This executes the generated code, calls generate(seed), validates the
    returned dataset, and sends the result back through the queue.
    """
    start_time = time.perf_counter()
    evaluator = DatasetEvaluator()

    def evaluate(X: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
        """
        Evaluate a candidate dataset under the fixed harness.

        Exposed to generated code so it can perform internal search.
        """
        return evaluator.evaluate_to_dict(X, y)

    safe_globals = {
        "__builtins__": _build_safe_builtins(),
        "np": np,
        "evaluate": evaluate,
    }
    local_vars: dict[str, object] = {}

    try:
        exec(python_code, safe_globals, local_vars)

        generate_fn = local_vars.get("generate", safe_globals.get("generate"))

        if generate_fn is None:
            raise ValueError("Code did not define a generate(seed) function")

        if not isinstance(generate_fn, FunctionType):
            raise ValueError("generate exists but is not a function")

        output = generate_fn(seed)

        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("generate(seed) must return a tuple (X, y)")

        X, y = output
        X, y = validate_dataset(X, y)

        runtime_seconds = time.perf_counter() - start_time

        result_queue.put(
            {
                "success": True,
                "X": X,
                "y": y,
                "error_type": None,
                "error_message": None,
                "runtime_seconds": runtime_seconds,
            }
        )

    except Exception as exc:
        runtime_seconds = time.perf_counter() - start_time

        result_queue.put(
            {
                "success": False,
                "X": None,
                "y": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "runtime_seconds": runtime_seconds,
            }
        )


class CodeRunner:
    """
    Execute generated code and call generate(seed).

    Expected contract:
    - the code defines a function named generate
    - generate accepts a single integer seed argument
    - generate returns (X, y)

    The generated code is also given access to:
    - np
    - evaluate(X, y): evaluates a candidate dataset under the fixed harness

    This allows generated code to perform internal search or tuning.

    The generate(seed) call is capped by timeout_seconds.
    """

    def __init__(self, timeout_seconds: int) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self.timeout_seconds = timeout_seconds

    def run(self, python_code: str, seed: int) -> ExecutionResult:
        """
        Execute the provided code and call generate(seed).

        If execution exceeds timeout_seconds, the worker process is terminated
        and a TimeoutError-style ExecutionResult is returned.
        """
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        process = ctx.Process(
            target=_worker_run_generated_code,
            args=(python_code, seed, result_queue),
        )

        start_time = time.perf_counter()
        process.start()

        try:
            payload: dict[str, Any] = result_queue.get(timeout=self.timeout_seconds)
            process.join(timeout=1.0)

            return ExecutionResult(
                success=bool(payload["success"]),
                X=payload["X"],
                y=payload["y"],
                error_type=payload["error_type"],
                error_message=payload["error_message"],
                runtime_seconds=float(payload["runtime_seconds"]),
            )

        except queue.Empty:
            process.terminate()
            process.join(timeout=1.0)

            runtime_seconds = time.perf_counter() - start_time

            return ExecutionResult(
                success=False,
                X=None,
                y=None,
                error_type="TimeoutError",
                error_message=(
                    f"Generated code execution exceeded timeout of "
                    f"{self.timeout_seconds} seconds"
                ),
                runtime_seconds=runtime_seconds,
            )

        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)