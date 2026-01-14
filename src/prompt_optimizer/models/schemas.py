"""
Pydantic schemas for Prompt Optimizer.

This module defines data models for:
- PII entity representation
- API responses (structured outputs)
- Error feedback for mentor model
- Optimization tracking
"""

from typing import Optional

from pydantic import BaseModel, Field


class PIIEntity(BaseModel):
    """
    Represents a single PII (Personally Identifiable Information) entity.

    Attributes:
        value: The actual PII value found in text.
        label: The category/type of PII (e.g., EMAIL, PHONE, FIRSTNAME).
        start: Starting character position in source text.
        end: Ending character position in source text.

    Examples:
        >>> entity = PIIEntity(
        ...     value="john@example.com",
        ...     label="EMAIL",
        ...     start=10,
        ...     end=26
        ... )
        >>> entity.label
        'EMAIL'
    """

    value: str = Field(..., description="The actual PII value found in text")
    label: str = Field(..., description="The category/type of PII")
    start: int = Field(..., description="Starting character position")
    end: int = Field(..., description="Ending character position")


class PIIResponse(BaseModel):
    """
    Structured response from the agent model containing extracted PII entities.

    Attributes:
        entities: List of PII entities found in the text.
        masked_text: The text with PII values replaced by labels.

    Examples:
        >>> response = PIIResponse(
        ...     entities=[
        ...         PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)
        ...     ],
        ...     masked_text="[FIRSTNAME] works at Acme Inc."
        ... )
        >>> len(response.entities)
        1
    """

    entities: list[PIIEntity] = Field(
        default_factory=list,
        description="List of PII entities found in the text",
    )
    masked_text: str = Field(
        default="",
        description="The text with PII values replaced by labels",
    )


class ErrorFeedback(BaseModel):
    """
    Error data sent to mentor model for analysis.

    Attributes:
        field_name: The PII field/label that was incorrectly identified.
        prompt: The prompt used for extraction.
        source_text: The original input text.
        agent_answer: The agent's incorrect response.
        ground_truth: The correct expected response.

    Examples:
        >>> error = ErrorFeedback(
        ...     field_name="EMAIL",
        ...     prompt="Extract PII from text",
        ...     source_text="Contact me at test@email.com",
        ...     agent_answer="No email found",
        ...     ground_truth="test@email.com"
        ... )
        >>> error.field_name
        'EMAIL'
    """

    field_name: str = Field(..., description="The PII field that was incorrect")
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
        ...     prompt="Extract all PII entities",
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
        ...     prompt="Extract PII",
        ...     accuracy=0.75,
        ...     error_summary={"EMAIL": 2, "PHONE": 1}
        ... )
        >>> history.error_summary["EMAIL"]
        2
    """

    iteration: int = Field(..., description="Iteration number")
    prompt: str = Field(..., description="The prompt text")
    accuracy: float = Field(..., description="Accuracy achieved")
    error_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of errors per field type",
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
        ...     current_prompt="Extract PII",
        ...     errors=[],
        ...     history=[],
        ...     schema_description="JSON with entities array"
        ... )
        >>> request.current_prompt
        'Extract PII'
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


class GeneratedPrompt(BaseModel):
    """
    Response from mentor model with generated prompt.

    Attributes:
        prompt: The generated or improved prompt text.
        reasoning: Explanation of changes made.

    Examples:
        >>> generated = GeneratedPrompt(
        ...     prompt="Extract all PII entities including names, emails...",
        ...     reasoning="Added more specific entity types"
        ... )
        >>> "PII" in generated.prompt
        True
    """

    prompt: str = Field(..., description="The generated prompt text")
    reasoning: str = Field(
        default="",
        description="Explanation of changes made",
    )
