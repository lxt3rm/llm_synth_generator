"""
Utilities for parsing structured LLM responses into project dataclasses.
"""

from __future__ import annotations

import json

from src.core.types import GeneratorResponse


EXPECTED_GENERATOR_RESPONSE_KEYS = {
    "mechanism_brief",
    "python_code",
    "expected_x_behavior",
    "expected_y_behavior",
}


def parse_generator_response(
    response_text: str,
    response_id: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> GeneratorResponse:
    """
    Parse a structured JSON string into GeneratorResponse.
    """
    data = json.loads(response_text)

    if not isinstance(data, dict):
        raise ValueError("Generator response must be a JSON object")

    actual_keys = set(data.keys())
    if actual_keys != EXPECTED_GENERATOR_RESPONSE_KEYS:
        missing = sorted(EXPECTED_GENERATOR_RESPONSE_KEYS - actual_keys)
        extra = sorted(actual_keys - EXPECTED_GENERATOR_RESPONSE_KEYS)
        raise ValueError(
            "Generator response JSON keys did not match the expected schema. "
            f"Missing keys: {missing}. Extra keys: {extra}."
        )

    return GeneratorResponse(
        mechanism_brief=data["mechanism_brief"],
        python_code=data["python_code"],
        expected_x_behavior=data["expected_x_behavior"],
        expected_y_behavior=data["expected_y_behavior"],
        raw_text=response_text,
        response_id=response_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
