"""
Wrapper around the OpenAI Responses API for generator creation.
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from src.core.types import GeneratorResponse, ModelConfig
from src.llm.response_parser import parse_generator_response
from src.llm.schemas import GENERATOR_RESPONSE_SCHEMA


def _safe_getattr(obj: Any, name: str, default=None):
    """Safely read an attribute from an SDK object."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _extract_token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """
    Extract token counts from a Responses API result if available.

    Returns:
        (input_tokens, output_tokens, total_tokens)

    This is intentionally defensive because SDK response shapes can vary.
    """
    usage = _safe_getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    input_tokens = _safe_getattr(usage, "input_tokens", None)
    output_tokens = _safe_getattr(usage, "output_tokens", None)
    total_tokens = _safe_getattr(usage, "total_tokens", None)

    # Fallback: compute total if only input/output are present.
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return input_tokens, output_tokens, total_tokens


class OpenAIGeneratorClient:
    """
    Thin wrapper around the OpenAI Responses API.

    This client:
    - sends prompts
    - requests structured JSON output
    - measures API runtime
    - parses the result into GeneratorResponse
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment")

        self.client = OpenAI(api_key=api_key)

    def generate_initial(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[GeneratorResponse, float]:
        """
        Make an initial generation call.

        Returns:
            (parsed_response, llm_runtime_seconds)
        """
        start_time = time.perf_counter()

        response = self.client.responses.create(
            model=self.model_config.name,
            instructions=system_prompt,
            input=user_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "dataset_generator_response",
                    "schema": GENERATOR_RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
            max_output_tokens=self.model_config.max_output_tokens,
        )

        runtime_seconds = time.perf_counter() - start_time
        input_tokens, output_tokens, total_tokens = _extract_token_usage(response)

        parsed = parse_generator_response(
            response.output_text,
            response_id=response.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        return parsed, runtime_seconds

    def generate_repair(
        self,
        system_prompt: str,
        user_prompt: str,
        previous_response_id: str,
    ) -> tuple[GeneratorResponse, float]:
        """
        Make a repair call continuing from a previous response state.

        Returns:
            (parsed_response, llm_runtime_seconds)
        """
        start_time = time.perf_counter()

        response = self.client.responses.create(
            model=self.model_config.name,
            instructions=system_prompt,
            input=user_prompt,
            previous_response_id=previous_response_id,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "dataset_generator_response",
                    "schema": GENERATOR_RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
            max_output_tokens=self.model_config.max_output_tokens,
        )

        runtime_seconds = time.perf_counter() - start_time
        input_tokens, output_tokens, total_tokens = _extract_token_usage(response)

        parsed = parse_generator_response(
            response.output_text,
            response_id=response.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        return parsed, runtime_seconds