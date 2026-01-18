"""
Unit tests for Pydantic models.
"""

import pytest

from prompt_optimizer.models import (
    PIIEntity,
    PIIResponse,
    ErrorFeedback,
    OptimizationResult,
    PromptHistory,
    MentorPromptRequest,
    GeneratedPrompt,
)


class TestPIIEntity:
    """Tests for PIIEntity model."""

    def test_create_entity(self) -> None:
        """Test creating a PIIEntity."""
        entity = PIIEntity(
            value="john@email.com",
            label="EMAIL",
            start=10,
            end=24,
        )
        
        assert entity.value == "john@email.com"
        assert entity.label == "EMAIL"
        assert entity.start == 10
        assert entity.end == 24

    def test_entity_serialization(self) -> None:
        """Test entity can be serialized to dict."""
        entity = PIIEntity(
            value="John",
            label="FIRSTNAME",
            start=0,
            end=4,
        )
        
        data = entity.model_dump()
        
        assert data["value"] == "John"
        assert data["label"] == "FIRSTNAME"


class TestPIIResponse:
    """Tests for PIIResponse model."""

    def test_create_response(self, sample_pii_entity: PIIEntity) -> None:
        """Test creating a PIIResponse."""
        response = PIIResponse(
            entities=[sample_pii_entity],
            masked_text="Contact: [EMAIL]",
        )
        
        assert len(response.entities) == 1
        assert response.masked_text == "Contact: [EMAIL]"

    def test_empty_response(self) -> None:
        """Test creating empty response."""
        response = PIIResponse()
        
        assert len(response.entities) == 0
        assert response.masked_text == ""

    def test_multiple_entities(self) -> None:
        """Test response with multiple entities."""
        entities = [
            PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
            PIIEntity(value="test@email.com", label="EMAIL", start=6, end=20),
        ]
        
        response = PIIResponse(
            entities=entities,
            masked_text="[FIRSTNAME] [EMAIL]",
        )
        
        assert len(response.entities) == 2
        assert response.entities[0].label == "FIRSTNAME"
        assert response.entities[1].label == "EMAIL"


class TestErrorFeedback:
    """Tests for ErrorFeedback model."""

    def test_create_error_feedback(self) -> None:
        """Test creating an ErrorFeedback."""
        error = ErrorFeedback(
            field_name="EMAIL",
            prompt="Extract all PII",
            source_text="Contact john@email.com",
            agent_answer="No email found",
            ground_truth="john@email.com",
        )
        
        assert error.field_name == "EMAIL"
        assert "Extract" in error.prompt
        assert error.agent_answer == "No email found"


class TestOptimizationResult:
    """Tests for OptimizationResult model."""

    def test_create_result(self) -> None:
        """Test creating an OptimizationResult."""
        result = OptimizationResult(
            iteration=1,
            prompt="Extract PII",
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
            field_name="EMAIL",
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
        assert result.errors[0].field_name == "EMAIL"


class TestPromptHistory:
    """Tests for PromptHistory model."""

    def test_create_history(self) -> None:
        """Test creating a PromptHistory."""
        history = PromptHistory(
            iteration=1,
            prompt="Extract PII entities",
            accuracy=0.75,
            error_summary={"EMAIL": 2, "PHONE": 1},
        )
        
        assert history.iteration == 1
        assert history.accuracy == 0.75
        assert history.error_summary["EMAIL"] == 2


class TestMentorPromptRequest:
    """Tests for MentorPromptRequest model."""

    def test_create_request(self) -> None:
        """Test creating a MentorPromptRequest."""
        request = MentorPromptRequest(
            current_prompt="Extract PII",
            errors=[],
            history=[],
            schema_description="JSON with entities",
        )
        
        assert request.current_prompt == "Extract PII"
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
            prompt="Extract all PII entities including names and emails",
            reasoning="Added more specific entity types",
        )
        
        assert "PII" in generated.prompt
        assert len(generated.reasoning) > 0
