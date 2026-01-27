"""
Unit tests for schema utilities.
"""

import json

import pytest

from prompt_optimizer.utils.schema import (
    export_schema,
    get_response_schema_description,
    get_extraction_schema,
)
from prompt_optimizer.models import ExtractionSchema


class TestExportSchema:
    """Tests for export_schema function."""

    def test_export_extraction_schema(self) -> None:
        """Test exporting ExtractionSchema."""
        schema = export_schema(ExtractionSchema)
        
        assert isinstance(schema, str)
        parsed = json.loads(schema)
        assert "properties" in parsed
        assert "firstname" in parsed["properties"]
        assert "email" in parsed["properties"]

    def test_export_schema_with_indent(self) -> None:
        """Test exporting schema with custom indent."""
        schema = export_schema(ExtractionSchema, indent=4)
        
        parsed = json.loads(schema)
        assert "properties" in parsed
        assert "amount" in parsed["properties"]


class TestGetResponseSchemaDescription:
    """Tests for get_response_schema_description function."""

    def test_description_contains_json_format(self) -> None:
        """Test that description mentions JSON format."""
        desc = get_response_schema_description()
        
        assert "JSON" in desc

    def test_description_contains_example(self) -> None:
        """Test that description includes an example."""
        desc = get_response_schema_description()
        
        assert "firstname" in desc
        assert "email" in desc


class TestGetExtractionSchema:
    """Tests for get_extraction_schema function."""

    def test_get_extraction_schema(self) -> None:
        """Test getting ExtractionSchema as dict."""
        schema = get_extraction_schema(ExtractionSchema)
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "firstname" in schema["properties"]
        assert "email" in schema["properties"]
        assert "amount" in schema["properties"]

    def test_schema_has_descriptions(self) -> None:
        """Test that schema fields have descriptions."""
        schema = get_extraction_schema(ExtractionSchema)
        
        firstname_prop = schema["properties"]["firstname"]
        assert "description" in firstname_prop
        assert "first name" in firstname_prop["description"].lower()

