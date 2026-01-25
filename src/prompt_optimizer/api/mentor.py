"""
Mentor model handler for generating and improving prompts.

The mentor model analyzes errors and historical performance to
generate improved prompts for the agent model. It also updates
field descriptions in the schema to improve extraction accuracy.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_optimizer.api.client import OpenRouterClient
from prompt_optimizer.config import get_settings
from prompt_optimizer.models import GeneratedPrompt
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

# Default log file path for mentor prompts
MENTOR_LOG_FILE = Path("mentor_prompts.txt")


class FailedPrediction:
    """Container for a failed prediction with source, ground truth, and prediction."""
    
    def __init__(
        self,
        source_text: str,
        ground_truth: dict[str, Any],
        predict: dict[str, Any],
    ):
        self.source_text = source_text
        self.ground_truth = ground_truth
        self.predict = predict
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "ground_truth": self.ground_truth,
            "predict": self.predict,
        }


class IterationHistory:
    """History entry for a single optimization iteration."""
    
    def __init__(
        self,
        iteration: int,
        prompt: str,
        prompt_accuracy: float,
        failed_predictions: list[FailedPrediction],
    ):
        self.iteration = iteration
        self.prompt = prompt
        self.prompt_accuracy = prompt_accuracy
        self.failed_predictions = failed_predictions
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "prompt": self.prompt,
            "prompt_accuracy": self.prompt_accuracy,
            "failed_predictions": [fp.to_dict() for fp in self.failed_predictions],
        }


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
        ...     history=[],
        ...     current_prompt="Extract key data"
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
            log_file: Path to log file for mentor prompts.
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
        ground_truth: dict[str, Any],
        current_field_descriptions: dict[str, str] | None = None,
    ) -> GeneratedPrompt:
        """
        Generate an initial prompt when no starting prompt is provided.

        The mentor receives sample data and ground truth, then creates
        a professional extraction prompt.

        Args:
            sample_data: Example input text for the task.
            ground_truth: Expected output as a dictionary.
            current_field_descriptions: Current field descriptions to start with.

        Returns:
            GeneratedPrompt: Generated prompt with reasoning and field descriptions.

        Examples:
            >>> mentor = MentorModel()
            >>> result = mentor.generate_initial_prompt(
            ...     sample_data="Contact John at john@email.com",
            ...     ground_truth={"firstname": "John", "email": "john@email.com"}
            ... )
            >>> len(result.prompt) > 0
            True
        """
        logger.info("=" * 60)
        logger.info("SEND MESSAGE TO MENTOR")
        logger.info("=" * 60)
        logger.info("Generating initial prompt from sample data and ground truth")
        
        system_message = """You are an expert prompt engineer.

Your goal is to create a clear, detailed prompt that will guide an AI assistant 
to accurately extract structured information from text.

The prompt should:
1. Be specific about what types of data to look for
2. Provide clear instructions on the output format (JSON)
3. Include guidance on edge cases
4. Be concise but comprehensive

You must also provide descriptions for each field type to help
the AI better understand what to extract."""

        field_desc_section = ""
        if current_field_descriptions:
            field_desc_section = f"""

CURRENT FIELD DESCRIPTIONS (improve these):
{self._format_field_descriptions(current_field_descriptions)}
"""

        import json
        ground_truth_str = json.dumps(ground_truth, indent=2)

        user_message = f"""Create a professional prompt for data extraction based on this example:

SAMPLE INPUT TEXT:
{sample_data}

EXPECTED OUTPUT (ground truth):
{ground_truth_str}
{field_desc_section}

Generate a prompt that would produce this expected output when given the sample input.
Also provide descriptions for each field type that appears in the expected output.
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
            reasoning_effort="low",
        )
        
        logger.info(f"Generated initial prompt: {response.prompt[:100]}...")
        if response.field_descriptions:
            logger.info(f"Updated field descriptions: {list(response.field_descriptions.keys())}")
        return response

    def generate_prompt(
        self,
        history: list[IterationHistory],
        current_prompt: str,
        current_field_descriptions: dict[str, str] | None = None,
    ) -> GeneratedPrompt:
        """
        Generate an improved prompt based on iteration history.

        Args:
            history: List of iteration histories with failed predictions.
            current_prompt: The current prompt being used.
            current_field_descriptions: Current field descriptions to improve.

        Returns:
            GeneratedPrompt: Improved prompt with reasoning and updated field descriptions.

        Examples:
            >>> mentor = MentorModel()
            >>> result = mentor.generate_prompt(
            ...     history=[],
            ...     current_prompt="Extract data"
            ... )
            >>> len(result.prompt) > 0
            True
        """
        logger.info("=" * 60)
        logger.info("SEND MESSAGE TO MENTOR")
        logger.info("=" * 60)
        logger.info(f"Generating improved prompt with {len(history)} history entries")
        
        system_message = """You are an expert prompt engineer analyzing extraction errors.

Your task is to improve the current prompt based on:
1. The agent's incorrect predictions shown in the history
2. Patterns in what the agent is missing or misidentifying
3. Accuracy trends across iterations

Generate an improved prompt that addresses these issues.

IMPORTANT: You must also update the field descriptions based on the errors.
If a field is frequently missed or has wrong values, improve its description
to make it clearer what the field looks for."""

        # Build history text
        history_text = self._format_history(history)
        
        field_desc_section = ""
        if current_field_descriptions:
            field_desc_section = f"""

CURRENT FIELD DESCRIPTIONS (update these based on errors):
{self._format_field_descriptions(current_field_descriptions)}
"""

        user_message = f"""Analyze the agent's predictions and improve the prompt:

CURRENT PROMPT:
{current_prompt}

ITERATION HISTORY (showing agent predictions vs ground truth):
{history_text}
{field_desc_section}

The agent's predictions are shown above. Improve the prompt to help the agent
produce better extractions. Be specific about what changes you're making and why."""

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
            reasoning_effort="low",
        )
        
        logger.info(f"Generated improved prompt: {response.prompt[:100]}...")
        if response.field_descriptions:
            logger.info(f"Updated field descriptions: {list(response.field_descriptions.keys())}")
        return response

    def _format_history(self, history: list[IterationHistory]) -> str:
        """
        Format iteration history for the mentor prompt.

        Args:
            history: List of iteration history entries.

        Returns:
            str: Formatted history text.
        """
        import json
        
        if not history:
            return "No previous iterations."
        
        lines = []
        for entry in history:
            lines.append(f"""
--- History {entry.iteration} ---
<prompt>
{entry.prompt}
</prompt>
prompt_accuracy: {entry.prompt_accuracy:.1f}%

Failed predictions:""")
            
            for fp in entry.failed_predictions:
                lines.append(f"""
[
  "source_text": "{fp.source_text[:200]}{'...' if len(fp.source_text) > 200 else ''}",
  "ground_truth": {json.dumps(fp.ground_truth)},
  "predict": {json.dumps(fp.predict)}
]""")
        
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
