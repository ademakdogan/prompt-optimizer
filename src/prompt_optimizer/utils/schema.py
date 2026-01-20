"""
Schema export utilities for Prompt Optimizer.

This module provides utilities for exporting Pydantic models
to JSON schema format for use in prompts.
"""

import json
from typing import Type

from pydantic import BaseModel

from prompt_optimizer.models import PIIResponse, PIIEntity
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
        >>> from prompt_optimizer.models import PIIResponse
        >>> schema = export_schema(PIIResponse)
        >>> "properties" in schema
        True
    """
    schema = model.model_json_schema()
    return json.dumps(schema, indent=indent)


def get_response_schema_description() -> str:
    """
    Get a human-readable description of the PIIResponse schema.

    Returns:
        str: Schema description for use in prompts.

    Examples:
        >>> desc = get_response_schema_description()
        >>> "entities" in desc
        True
    """
    return """
The response should be a JSON object with the following structure:
{
    "entities": [
        {
            "value": "The actual PII text found",
            "label": "The PII type (e.g., EMAIL, FIRSTNAME, PHONE)",
            "start": 0,  // Starting character position
            "end": 10    // Ending character position
        }
    ],
    "masked_text": "The input text with PII replaced by [LABEL] placeholders"
}

Available PII types include: FIRSTNAME, LASTNAME, EMAIL, PHONENUMBER, 
ADDRESS, SSN, CREDITCARDNUMBER, DATE, TIME, URL, IP, USERNAME, PASSWORD,
NEARBYGPSCOORDINATE, USERAGENT, JOBTITLE, COUNTY, ACCOUNTNUMBER, etc.
"""


def get_pii_entity_schema() -> dict:
    """
    Get the JSON schema for PIIEntity.

    Returns:
        dict: JSON schema dictionary.

    Examples:
        >>> schema = get_pii_entity_schema()
        >>> schema["type"] == "object"
        True
    """
    return PIIEntity.model_json_schema()


def get_pii_response_schema() -> dict:
    """
    Get the JSON schema for PIIResponse.

    Returns:
        dict: JSON schema dictionary.

    Examples:
        >>> schema = get_pii_response_schema()
        >>> "entities" in schema.get("properties", {})
        True
    """
    return PIIResponse.model_json_schema()
