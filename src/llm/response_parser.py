"""
Utilities for parsing structured LLM responses into project dataclasses.
"""

from __future__ import annotations

import json

from src.core.types import GeneratorResponse


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