"""
Pydantic schemas for Prompt Optimizer.

This module defines data models for:
- Error feedback for mentor model
- Optimization tracking
- Generated prompt responses
"""

from typing import Optional, Any

from pydantic import BaseModel, Field


class ErrorFeedback(BaseModel):
    """
    Error data sent to mentor model for analysis.

    Attributes:
        field_name: The field that was incorrectly identified.
        prompt: The prompt used for extraction.
        source_text: The original input text.
        agent_answer: The agent's incorrect response.
        ground_truth: The correct expected response.

    Examples:
        >>> error = ErrorFeedback(
        ...     field_name="email",
        ...     prompt="Extract data from text",
        ...     source_text="Contact me at test@email.com",
        ...     agent_answer="No email found",
        ...     ground_truth="test@email.com"
        ... )
        >>> error.field_name
        'email'
    """

    field_name: str = Field(..., description="The field that was incorrect")
    prompt: str = Field(..., description="The prompt used for extraction")
    source_text: str = Field(..., description="The original input text")
    agent_answer: str = Field(..., description="The agent's response")
    ground_truth: str = Field(..., description="The correct expected value")


class OptimizationResult(BaseModel):
    """
    Result of a single optimization iteration.

    Attributes:
        iteration: The iteration number (1-indexed).
        prompt: The prompt used in this iteration.
        accuracy: Accuracy score (0.0 to 1.0).
        total_samples: Total number of samples processed.
        correct_samples: Number of correctly processed samples.
        errors: List of errors found in this iteration.

    Examples:
        >>> result = OptimizationResult(
        ...     iteration=1,
        ...     prompt="Extract all data fields",
        ...     accuracy=0.85,
        ...     total_samples=10,
        ...     correct_samples=8.5,
        ...     errors=[]
        ... )
        >>> result.accuracy
        0.85
    """

    iteration: int = Field(..., description="Iteration number (1-indexed)")
    prompt: str = Field(..., description="The prompt used")
    accuracy: float = Field(..., description="Accuracy score (0.0 to 1.0)")
    total_samples: int = Field(..., description="Total samples processed")
    correct_samples: float = Field(..., description="Number of correct samples")
    errors: list[ErrorFeedback] = Field(
        default_factory=list,
        description="Errors found in this iteration",
    )
    field_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Updated field descriptions from mentor",
    )


class PromptHistory(BaseModel):
    """
    Track prompt versions and their performance for mentor context.

    Attributes:
        iteration: The iteration number.
        prompt: The prompt text.
        accuracy: Accuracy achieved with this prompt.
        error_summary: Summary of unique error types.

    Examples:
        >>> history = PromptHistory(
        ...     iteration=1,
        ...     prompt="Extract data",
        ...     accuracy=0.75,
        ...     error_summary={"email": 2, "phone": 1}
        ... )
        >>> history.error_summary["email"]
        2
    """

    iteration: int = Field(..., description="Iteration number")
    prompt: str = Field(..., description="The prompt text")
    accuracy: float = Field(..., description="Accuracy achieved")
    error_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of errors per field type",
    )
    field_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Field descriptions used in this iteration",
    )


class MentorPromptRequest(BaseModel):
    """
    Request to mentor model for generating improved prompt.

    Attributes:
        current_prompt: The current prompt being used.
        errors: List of error feedback from current iteration.
        history: Historical prompt performance (limited by window_size).
        schema_description: Description of expected output schema.

    Examples:
        >>> request = MentorPromptRequest(
        ...     current_prompt="Extract data",
        ...     errors=[],
        ...     history=[],
        ...     schema_description="JSON with key-value pairs"
        ... )
        >>> request.current_prompt
        'Extract data'
    """

    current_prompt: Optional[str] = Field(
        None,
        description="The current prompt (None for initial generation)",
    )
    errors: list[ErrorFeedback] = Field(
        default_factory=list,
        description="Error feedback from current iteration",
    )
    history: list[PromptHistory] = Field(
        default_factory=list,
        description="Historical prompt performance",
    )
    schema_description: str = Field(
        default="",
        description="Description of expected output schema",
    )
    current_field_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Current field descriptions to improve",
    )


class GeneratedPrompt(BaseModel):
    """
    Response from mentor model with generated prompt and field descriptions.

    Attributes:
        prompt: The generated or improved prompt text.
        reasoning: Explanation of changes made.
        field_descriptions: Updated field descriptions for schema.

    Examples:
        >>> generated = GeneratedPrompt(
        ...     prompt="Extract all key information including names, emails...",
        ...     reasoning="Added more specific field types",
        ...     field_descriptions={"email": "Email in format user@domain.com"}
        ... )
        >>> "Extract" in generated.prompt
        True
    """

    prompt: str = Field(..., description="The generated prompt text")
    reasoning: str = Field(
        default="",
        description="Explanation of changes made",
    )
    field_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Updated descriptions for each field to improve extraction",
    )


# Legacy models kept for backward compatibility
class PIIEntity(BaseModel):
    """Legacy model - kept for backward compatibility."""
    value: str = Field(..., description="The value found in text")
    label: str = Field(..., description="The category/type")
    start: int = Field(..., description="Starting character position")
    end: int = Field(..., description="Ending character position")


class PIIResponse(BaseModel):
    """Legacy model - kept for backward compatibility."""
    entities: list[PIIEntity] = Field(default_factory=list)
    masked_text: str = Field(default="")
