"""
Mentor model handler for generating and improving prompts.

The mentor model analyzes errors and historical performance to
generate improved prompts for the agent model. It also updates
field descriptions in the schema to improve extraction accuracy.
"""

from datetime import datetime
from pathlib import Path

from prompt_optimizer.api.client import OpenRouterClient
from prompt_optimizer.config import get_settings
from prompt_optimizer.models import (
    ErrorFeedback,
    GeneratedPrompt,
    PromptHistory,
)
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

# Default log file path for mentor prompts
MENTOR_LOG_FILE = Path("mentor_prompts.txt")


class MentorModel:
    """
    Mentor model for generating and improving prompts.

    The mentor analyzes error patterns and historical performance
    to generate optimized prompts for the agent model. It also
    updates field descriptions to improve schema understanding.

    Attributes:
        client: The OpenRouter client instance.
        log_file: Path to the file for logging mentor prompts.

    Examples:
        >>> mentor = MentorModel()
        >>> result = mentor.generate_prompt(
        ...     errors=[],
        ...     history=[],
        ...     current_prompt="Extract PII"
        ... )
        >>> isinstance(result, GeneratedPrompt)
        True
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        """
        Initialize the mentor model.

        Args:
            model: Model identifier. If None, uses settings.mentor_model.
            api_key: OpenRouter API key. If None, loaded from settings.
            log_file: Path to log file for mentor prompts. Defaults to mentor_prompts.txt
        """
        settings = get_settings()
        model_name = model or settings.mentor_model
        
        self.client = OpenRouterClient(model=model_name, api_key=api_key)
        self.log_file = Path(log_file) if log_file else MENTOR_LOG_FILE
        
        logger.info(f"Initialized MentorModel with model: {model_name}")
        logger.info(f"Mentor prompts will be logged to: {self.log_file}")

    def _log_mentor_prompt(self, prompt_type: str, messages: list[dict[str, str]]) -> None:
        """
        Log mentor prompt messages to a text file.

        Args:
            prompt_type: Type of prompt (e.g., "initial", "improved").
            messages: The messages being sent to the mentor.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"MENTOR PROMPT - {prompt_type.upper()}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*80}\n\n")
            
            for msg in messages:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                f.write(f"[{role}]\n")
                f.write(f"{content}\n\n")
            
            f.write(f"{'='*80}\n")

    def generate_initial_prompt(
        self,
        sample_data: str,
        ground_truth: str,
        schema_description: str = "",
        current_field_descriptions: dict[str, str] | None = None,
    ) -> GeneratedPrompt:
        """
        Generate an initial prompt when no starting prompt is provided.

        Args:
            sample_data: Example input data for the task.
            ground_truth: Expected output for the sample data.
            schema_description: Description of the expected output schema.
            current_field_descriptions: Current field descriptions to start with.

        Returns:
            GeneratedPrompt: Generated prompt with reasoning and field descriptions.

        Examples:
            >>> mentor = MentorModel()
            >>> result = mentor.generate_initial_prompt(
            ...     sample_data="Contact John at john@email.com",
            ...     ground_truth='{"firstname": "John", "email": "john@email.com"}'
            ... )
            >>> len(result.prompt) > 0
            True
        """
        logger.info("=" * 60)
        logger.info("SEND MESSAGE TO MENTOR")
        logger.info("=" * 60)
        logger.info("Generating initial prompt from sample data")
        
        system_message = """You are an expert prompt engineer specializing in PII extraction tasks.

Your goal is to create a clear, detailed prompt that will guide an AI assistant 
to accurately extract Personally Identifiable Information (PII) from text.

The prompt should:
1. Be specific about what types of PII to look for
2. Provide clear instructions on the output format
3. Include guidance on edge cases
4. Be concise but comprehensive

You must also provide improved descriptions for each field type to help
the AI better understand what to extract. These descriptions will be used
in the JSON schema to guide structured output parsing."""

        field_desc_section = ""
        if current_field_descriptions:
            field_desc_section = f"""

CURRENT FIELD DESCRIPTIONS (improve these):
{self._format_field_descriptions(current_field_descriptions)}
"""

        user_message = f"""Create a professional prompt for PII extraction based on this example:

SAMPLE INPUT:
{sample_data}

EXPECTED OUTPUT (target_result as JSON):
{ground_truth}

{f"OUTPUT SCHEMA: {schema_description}" if schema_description else ""}
{field_desc_section}

Generate a prompt that would produce this expected output when given the sample input.
Also provide improved descriptions for each field type that appears in the expected output.
These descriptions should help the AI understand exactly what to look for.

Explain your reasoning for the prompt design."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        
        # Log to file
        self._log_mentor_prompt("initial", messages)
        
        logger.info(f"User message to mentor:\n{user_message[:500]}...")
        
        response = self.client.chat(
            messages=messages,
            response_model=GeneratedPrompt,
            temperature=0.7,
            reasoning_effort="low",  # Minimize thinking tokens
        )
        
        logger.info(f"Generated initial prompt: {response.prompt[:100]}...")
        if response.field_descriptions:
            logger.info(f"Updated field descriptions: {list(response.field_descriptions.keys())}")
        return response

    def generate_prompt(
        self,
        errors: list[ErrorFeedback],
        history: list[PromptHistory],
        current_prompt: str | None = None,
        schema_description: str = "",
        current_field_descriptions: dict[str, str] | None = None,
    ) -> GeneratedPrompt:
        """
        Generate an improved prompt based on errors and history.

        Args:
            errors: List of error feedback from the current iteration.
            history: Historical prompt performance (limited by window_size).
            current_prompt: The current prompt being used.
            schema_description: Description of the expected output schema.
            current_field_descriptions: Current field descriptions to improve.

        Returns:
            GeneratedPrompt: Improved prompt with reasoning and updated field descriptions.

        Examples:
            >>> mentor = MentorModel()
            >>> error = ErrorFeedback(
            ...     field_name="EMAIL",
            ...     prompt="Extract PII",
            ...     source_text="Contact test@mail.com",
            ...     agent_answer="No email found",
            ...     ground_truth="test@mail.com"
            ... )
            >>> result = mentor.generate_prompt(
            ...     errors=[error],
            ...     history=[],
            ...     current_prompt="Extract PII"
            ... )
            >>> len(result.prompt) > 0
            True
        """
        logger.info("=" * 60)
        logger.info("SEND MESSAGE TO MENTOR")
        logger.info("=" * 60)
        logger.info(f"Generating improved prompt with {len(errors)} errors")
        
        system_message = """You are an expert prompt engineer analyzing PII extraction errors.

Your task is to improve the current prompt based on:
1. Specific errors made by the agent
2. Historical performance of previous prompts
3. Patterns in what the agent is missing or misidentifying

Generate an improved prompt that addresses these issues while maintaining 
good performance on previously correct extractions.

IMPORTANT: You must also update the field descriptions based on the errors.
If a field is frequently missed or misidentified, improve its description
to make it clearer what the field looks for. These descriptions will be
used in the JSON schema for structured output parsing."""

        # Build error summary
        error_text = self._format_errors(errors)
        history_text = self._format_history(history)
        
        field_desc_section = ""
        if current_field_descriptions:
            field_desc_section = f"""

CURRENT FIELD DESCRIPTIONS (update these based on errors):
{self._format_field_descriptions(current_field_descriptions)}
"""

        user_message = f"""Analyze these extraction errors and improve the prompt:

CURRENT PROMPT:
{current_prompt or "No prompt provided - this is the first iteration"}

ERRORS FROM CURRENT ITERATION:
{error_text}

HISTORICAL PERFORMANCE:
{history_text}

{f"OUTPUT SCHEMA: {schema_description}" if schema_description else ""}
{field_desc_section}

Generate an improved prompt that addresses the identified issues.
Also update the field descriptions for any fields that have errors.
Be specific about what changes you're making and why."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        
        # Log to file
        self._log_mentor_prompt("improved", messages)
        
        logger.info(f"User message to mentor:\n{user_message[:500]}...")
        
        response = self.client.chat(
            messages=messages,
            response_model=GeneratedPrompt,
            temperature=0.7,
            reasoning_effort="low",  # Minimize thinking tokens
        )
        
        logger.info(f"Generated improved prompt: {response.prompt[:100]}...")
        if response.field_descriptions:
            logger.info(f"Updated field descriptions: {list(response.field_descriptions.keys())}")
        return response

    def _format_errors(self, errors: list[ErrorFeedback]) -> str:
        """
        Format error feedback for the mentor prompt.

        Args:
            errors: List of error feedback.

        Returns:
            str: Formatted error text.
        """
        if not errors:
            return "No errors in current iteration."
        
        lines = []
        for i, error in enumerate(errors, 1):
            lines.append(f"""
Error {i} (Field: {error.field_name}):
- Source text: {error.source_text[:200]}...
- Agent's answer: {error.agent_answer}
- Expected (ground truth): {error.ground_truth}
""")
        return "\n".join(lines)

    def _format_history(self, history: list[PromptHistory]) -> str:
        """
        Format historical performance for the mentor prompt.

        Args:
            history: List of prompt history entries.

        Returns:
            str: Formatted history text.
        """
        if not history:
            return "No previous iterations."
        
        lines = []
        for entry in history:
            error_counts = ", ".join(
                f"{k}: {v}" for k, v in entry.error_summary.items()
            )
            lines.append(f"""
Iteration {entry.iteration}:
- Accuracy: {entry.accuracy:.2%}
- Prompt: {entry.prompt[:100]}...
- Error types: {error_counts or "None"}
""")
        return "\n".join(lines)

    def _format_field_descriptions(self, descriptions: dict[str, str]) -> str:
        """
        Format field descriptions for the mentor prompt.

        Args:
            descriptions: Dict mapping field names to descriptions.

        Returns:
            str: Formatted field descriptions text.
        """
        if not descriptions:
            return "No field descriptions provided."
        
        lines = []
        for field, desc in descriptions.items():
            lines.append(f"- {field}: {desc}")
        return "\n".join(lines)
