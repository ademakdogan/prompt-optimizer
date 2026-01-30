"""
Schema export utilities for Prompt Optimizer.

This module provides utilities for exporting Pydantic models
to JSON schema format for use in prompts.
"""

import json
from typing import Type

from pydantic import BaseModel

from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


def export_schema(model: Type[BaseModel], indent: int = 2) -> str:
    """
    Export a Pydantic model to JSON schema string.

    Args:
        model: The Pydantic model class.
        indent: Indentation level for JSON formatting.

    Returns:
        str: JSON schema as formatted string.

    Examples:
        >>> from prompt_optimizer.models import ExtractionSchema
        >>> schema = export_schema(ExtractionSchema)
        >>> "properties" in schema
        True
    """
    schema = model.model_json_schema()
    return json.dumps(schema, indent=indent)


def get_response_schema_description() -> str:
    """
    Get a human-readable description of the expected response schema.

    Returns:
        str: Schema description for use in prompts.

    Examples:
        >>> desc = get_response_schema_description()
        >>> "JSON" in desc
        True
    """
    return """
The response should be a JSON object containing extracted field values.

Format: A flat JSON object where keys are field names and values are the 
extracted data. Only include fields that are found or can be calculated.

Example:
{
    "client_name": "TechSolutions Inc",
    "total_gross": 1180.0,
    "total_mid_gross": 1280.0
}

Rules:
- Only extract information explicitly present in the text
- Calculate derived fields based on provided formulas
- Omit fields that are not found or cannot be calculated
- Return exact values as they appear (preserve formatting)
- Return ONLY valid JSON, no additional text
"""


def get_extraction_schema(model: Type[BaseModel]) -> dict:
    """
    Get the JSON schema for an extraction model.

    Args:
        model: The Pydantic model class to get schema for.

    Returns:
        dict: JSON schema dictionary.

    Examples:
        >>> from prompt_optimizer.models import ExtractionSchema
        >>> schema = get_extraction_schema(ExtractionSchema)
        >>> schema["type"] == "object"
        True
    """
    return model.model_json_schema()

