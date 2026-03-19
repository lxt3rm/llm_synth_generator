"""
Structured output schemas for LLM responses.
"""

GENERATOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanism_brief": {
            "type": "string",
            "description": "Short explanation of the intended generation mechanism."
        },
        "python_code": {
            "type": "string",
            "description": "Python code defining generate(seed: int) -> tuple[np.ndarray, np.ndarray]."
        },
        "expected_x_behavior": {
            "type": "string",
            "description": "What behavior is expected along the x-axis metric."
        },
        "expected_y_behavior": {
            "type": "string",
            "description": "What behavior is expected along the y-axis metric."
        }
    },
    "required": [
        "mechanism_brief",
        "python_code",
        "expected_x_behavior",
        "expected_y_behavior"
    ],
    "additionalProperties": False
}