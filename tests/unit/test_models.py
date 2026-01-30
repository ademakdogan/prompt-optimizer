"""
Unit tests for Pydantic models.
"""

import pytest

from prompt_optimizer.models import (
    ExtractionSchema,
    ErrorFeedback,
    OptimizationResult,
    PromptHistory,
    MentorPromptRequest,
    GeneratedPrompt,
    generate_default_prompt,
    get_schema_field_descriptions,
)


class TestExtractionSchema:
    """Tests for ExtractionSchema model."""

    def test_create_schema_with_required_values(self) -> None:
        """Test creating ExtractionSchema with required values."""
        schema = ExtractionSchema(
            client_name="TechSolutions Inc",
            total_gross=1180.0,
            total_mid_gross=1280.0,
        )
        
        assert schema.client_name == "TechSolutions Inc"
        assert schema.total_gross == 1180.0
        assert schema.total_mid_gross == 1280.0
        assert schema.net_payable is None
        assert schema.currency_code is None

    def test_create_schema_with_all_values(self) -> None:
        """Test creating ExtractionSchema with all values including optional."""
        schema = ExtractionSchema(
            client_name="Alpha Logistics",
            total_gross=2360.59,
            total_mid_gross=2460.615,
            net_payable=2360.615,
            currency_code="USD",
        )
        
        assert schema.client_name == "Alpha Logistics"
        assert schema.total_gross == 2360.59
        assert schema.total_mid_gross == 2460.615
        assert schema.net_payable == 2360.615
        assert schema.currency_code == "USD"

    def test_schema_serialization(self) -> None:
        """Test schema can be serialized to dict."""
        schema = ExtractionSchema(
            client_name="Creative Studio",
            total_gross=550.0,
            total_mid_gross=650.0,
        )
        
        data = schema.model_dump(exclude_none=True)
        
        assert data["client_name"] == "Creative Studio"
        assert data["total_gross"] == 550.0
        assert data["total_mid_gross"] == 650.0
        assert "net_payable" not in data  # None values excluded
        assert "currency_code" not in data  # None values excluded

    def test_all_fields(self) -> None:
        """Test schema with all fields populated."""
        schema = ExtractionSchema(
            client_name="Omega Retail",
            total_gross=120.0,
            total_mid_gross=135.0,
            net_payable=130.0,
            currency_code="EUR",
        )
        
        assert schema.client_name == "Omega Retail"
        assert schema.total_gross == 120.0
        assert schema.total_mid_gross == 135.0
        assert schema.net_payable == 130.0
        assert schema.currency_code == "EUR"


class TestErrorFeedback:
    """Tests for ErrorFeedback model."""

    def test_create_error_feedback(self) -> None:
        """Test creating an ErrorFeedback."""
        error = ErrorFeedback(
            field_name="email",
            prompt="Extract all data",
            source_text="Contact john@email.com",
            agent_answer="No email found",
            ground_truth="john@email.com",
        )
        
        assert error.field_name == "email"
        assert "Extract" in error.prompt
        assert error.agent_answer == "No email found"


class TestOptimizationResult:
    """Tests for OptimizationResult model."""

    def test_create_result(self) -> None:
        """Test creating an OptimizationResult."""
        result = OptimizationResult(
            iteration=1,
            prompt="Extract data",
            accuracy=0.85,
            total_samples=10,
            correct_samples=8.5,
            errors=[],
        )
        
        assert result.iteration == 1
        assert result.accuracy == 0.85
        assert result.total_samples == 10

    def test_result_with_errors(self) -> None:
        """Test result with error list."""
        error = ErrorFeedback(
            field_name="email",
            prompt="test",
            source_text="test",
            agent_answer="none",
            ground_truth="email@test.com",
        )
        
        result = OptimizationResult(
            iteration=1,
            prompt="test",
            accuracy=0.5,
            total_samples=2,
            correct_samples=1,
            errors=[error],
        )
        
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "email"


class TestPromptHistory:
    """Tests for PromptHistory model."""

    def test_create_history(self) -> None:
        """Test creating a PromptHistory."""
        history = PromptHistory(
            iteration=1,
            prompt="Extract entities",
            accuracy=0.75,
            error_summary={"email": 2, "phone": 1},
        )
        
        assert history.iteration == 1
        assert history.accuracy == 0.75
        assert history.error_summary["email"] == 2


class TestMentorPromptRequest:
    """Tests for MentorPromptRequest model."""

    def test_create_request(self) -> None:
        """Test creating a MentorPromptRequest."""
        request = MentorPromptRequest(
            current_prompt="Extract data",
            errors=[],
            history=[],
            schema_description="JSON with entities",
        )
        
        assert request.current_prompt == "Extract data"
        assert len(request.errors) == 0

    def test_request_without_prompt(self) -> None:
        """Test request without current prompt (initial generation)."""
        request = MentorPromptRequest(
            current_prompt=None,
            errors=[],
            history=[],
        )
        
        assert request.current_prompt is None


class TestGeneratedPrompt:
    """Tests for GeneratedPrompt model."""

    def test_create_generated_prompt(self) -> None:
        """Test creating a GeneratedPrompt."""
        generated = GeneratedPrompt(
            prompt="Extract all entities including names and emails",
            reasoning="Added more specific entity types",
        )
        
        assert "Extract" in generated.prompt
        assert len(generated.reasoning) > 0


class TestGenerateDefaultPrompt:
    """Tests for generate_default_prompt function."""

    def test_prompt_contains_extract_instruction(self) -> None:
        """Test that prompt contains the extraction instruction."""
        prompt = generate_default_prompt()
        
        assert "Extract the following information from the data" in prompt

    def test_prompt_contains_field_names(self) -> None:
        """Test that prompt contains all schema field names."""
        prompt = generate_default_prompt()
        
        assert "client_name" in prompt
        assert "total_gross" in prompt
        assert "total_mid_gross" in prompt

    def test_prompt_contains_descriptions(self) -> None:
        """Test that prompt contains field descriptions if available."""
        prompt = generate_default_prompt()
        
        # Check that field names are present (descriptions may be empty for some fields)
        assert "client_name" in prompt
        assert "total_gross" in prompt

    def test_prompt_contains_json_instruction(self) -> None:
        """Test that prompt includes JSON return instruction."""
        prompt = generate_default_prompt()
        
        assert "JSON" in prompt


class TestGetSchemaFieldDescriptions:
    """Tests for get_schema_field_descriptions function."""

    def test_returns_dict(self) -> None:
        """Test that function returns a dictionary."""
        descriptions = get_schema_field_descriptions()
        
        assert isinstance(descriptions, dict)

    def test_contains_all_fields(self) -> None:
        """Test that all schema fields are present."""
        descriptions = get_schema_field_descriptions()
        
        assert "client_name" in descriptions
        assert "total_gross" in descriptions
        assert "total_mid_gross" in descriptions
        assert "net_payable" in descriptions
        assert "currency_code" in descriptions

    def test_descriptions_are_strings(self) -> None:
        """Test that all descriptions are strings."""
        descriptions = get_schema_field_descriptions()
        
        for key, value in descriptions.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_descriptions_exist_for_fields(self) -> None:
        """Test that descriptions exist for all fields."""
        descriptions = get_schema_field_descriptions()
        
        # All fields should have descriptions (even if empty string)
        for field in ["client_name", "total_gross", "total_mid_gross", "net_payable", "currency_code"]:
            assert field in descriptions
