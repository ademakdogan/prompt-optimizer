"""
Unit tests for schema utilities.
"""

import json

import pytest

from prompt_optimizer.utils.schema import (
    export_schema,
    get_response_schema_description,
    get_pii_entity_schema,
    get_pii_response_schema,
)
from prompt_optimizer.models import PIIEntity, PIIResponse


class TestExportSchema:
    """Tests for export_schema function."""

    def test_export_pii_entity_schema(self) -> None:
        """Test exporting PIIEntity schema."""
        schema = export_schema(PIIEntity)
        
        assert isinstance(schema, str)
        parsed = json.loads(schema)
        assert "properties" in parsed
        assert "value" in parsed["properties"]
        assert "label" in parsed["properties"]

    def test_export_pii_response_schema(self) -> None:
        """Test exporting PIIResponse schema."""
        schema = export_schema(PIIResponse)
        
        parsed = json.loads(schema)
        assert "properties" in parsed
        assert "entities" in parsed["properties"]
        assert "masked_text" in parsed["properties"]


class TestGetResponseSchemaDescription:
    """Tests for get_response_schema_description function."""

    def test_description_contains_entities(self) -> None:
        """Test that description mentions entities."""
        desc = get_response_schema_description()
        
        assert "entities" in desc
        assert "masked_text" in desc

    def test_description_contains_pii_types(self) -> None:
        """Test that description lists PII types."""
        desc = get_response_schema_description()
        
        assert "EMAIL" in desc
        assert "FIRSTNAME" in desc
        assert "PHONENUMBER" in desc


class TestGetPiiSchemas:
    """Tests for schema getter functions."""

    def test_get_pii_entity_schema(self) -> None:
        """Test getting PIIEntity schema."""
        schema = get_pii_entity_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_get_pii_response_schema(self) -> None:
        """Test getting PIIResponse schema."""
        schema = get_pii_response_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "entities" in schema["properties"]
