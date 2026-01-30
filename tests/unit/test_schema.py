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
        assert "client_name" in parsed["properties"]
        assert "total_gross" in parsed["properties"]

    def test_export_schema_with_indent(self) -> None:
        """Test exporting schema with custom indent."""
        schema = export_schema(ExtractionSchema, indent=4)
        
        parsed = json.loads(schema)
        assert "properties" in parsed
        assert "total_mid_gross" in parsed["properties"]


class TestGetResponseSchemaDescription:
    """Tests for get_response_schema_description function."""

    def test_description_contains_json_format(self) -> None:
        """Test that description mentions JSON format."""
        desc = get_response_schema_description()
        
        assert "JSON" in desc

    def test_description_contains_example(self) -> None:
        """Test that description includes field names."""
        desc = get_response_schema_description()
        
        assert "client_name" in desc
        assert "total_gross" in desc


class TestGetExtractionSchema:
    """Tests for get_extraction_schema function."""

    def test_get_extraction_schema(self) -> None:
        """Test getting ExtractionSchema as dict."""
        schema = get_extraction_schema(ExtractionSchema)
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "client_name" in schema["properties"]
        assert "total_gross" in schema["properties"]
        assert "total_mid_gross" in schema["properties"]

    def test_schema_has_required_fields(self) -> None:
        """Test that schema has required fields defined."""
        schema = get_extraction_schema(ExtractionSchema)
        
        # Check required fields are marked
        assert "required" in schema
        assert "client_name" in schema["required"]
        assert "total_gross" in schema["required"]

