"""
Agent model handler for processing data with prompts.

The agent model processes input data using a given prompt and
returns structured extraction results.
"""

from typing import Any

from prompt_optimizer.api.client import OpenRouterClient
from prompt_optimizer.config import get_settings
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class AgentModel:
    """
    Agent model for processing data with prompts.

    The agent takes a prompt and input data, then returns
    structured extraction results based on the schema.

    Attributes:
        client: The OpenRouter client instance.
        field_descriptions: Current field descriptions for schema.

    Examples:
        >>> agent = AgentModel()
        >>> response = agent.process_data(
        ...     prompt="Extract key information from the text.",
        ...     data="Contact John at john@email.com"
        ... )
        >>> isinstance(response, dict)
        True
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        """
        Initialize the agent model.

        Args:
            model: Model identifier. If None, uses settings.agent_model.
            api_key: OpenRouter API key. If None, loaded from settings.
        """
        settings = get_settings()
        model_name = model or settings.agent_model
        
        self.client = OpenRouterClient(model=model_name, api_key=api_key)
        self.field_descriptions: dict[str, str] = {}
        logger.info(f"Initialized AgentModel with model: {model_name}")

    def update_field_descriptions(self, descriptions: dict[str, str]) -> None:
        """
        Update field descriptions for improved extraction.

        Args:
            descriptions: Dict mapping field names to descriptions.

        Examples:
            >>> agent = AgentModel()
            >>> agent.update_field_descriptions({"email": "Email in user@domain format"})
        """
        self.field_descriptions.update(descriptions)
        logger.info(f"Updated field descriptions: {list(descriptions.keys())}")

    def process_data(
        self,
        prompt: str,
        data: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process input data using the given prompt.

        Args:
            prompt: The instruction prompt for data extraction.
            data: The input text to analyze.
            response_schema: Optional schema describing expected output fields.

        Returns:
            dict: Extracted data as key-value pairs.

        Examples:
            >>> agent = AgentModel()
            >>> result = agent.process_data(
            ...     prompt="Extract contact information",
            ...     data="Email: test@example.com"
            ... )
            >>> isinstance(result, dict)
            True
        """
        logger.debug(f"Processing data with prompt: {prompt[:50]}...")
        
        system_message = self._build_system_message(prompt, response_schema)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Extract information from this text:\n\n{data}"},
        ]
        
        # Use raw chat and parse JSON response
        response_text = self.client.chat_raw(
            messages=messages,
            temperature=0.3,
            reasoning_effort="low",
        )
        
        # Parse JSON from response
        result = self._parse_json_response(response_text)
        
        logger.debug(f"Extracted {len(result)} fields")
        return result

    def _build_system_message(
        self,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """
        Build the system message for the API request.

        Args:
            prompt: The user's extraction prompt.
            response_schema: Optional schema describing expected fields.

        Returns:
            str: The formatted system message.
        """
        # Build field descriptions section
        field_desc_section = ""
        if self.field_descriptions:
            lines = [f"- {name}: {desc}" for name, desc in self.field_descriptions.items()]
            field_desc_section = f"""

FIELD DESCRIPTIONS (use these to understand what to extract):
{chr(10).join(lines)}
"""

        schema_section = ""
        if response_schema:
            schema_section = f"""

EXPECTED OUTPUT SCHEMA:
{response_schema}
"""

        base_message = f"""You are a data extraction assistant.

{prompt}

Your task is to extract structured information from the given text and return it as a JSON object.

Rules:
1. Only extract information that is explicitly present in the text
2. Use the field names as specified in the schema or prompt
3. If a field is not found in the text, do not include it in the output
4. Return ONLY valid JSON, no additional text or explanation
{field_desc_section}{schema_section}
Be thorough and accurate in your extraction."""

        return base_message

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """
        Parse JSON from the model response.

        Args:
            response_text: Raw text response from the model.

        Returns:
            dict: Parsed JSON data.
        """
        import json
        import re
        
        # Try to extract JSON from the response
        # First, try to find JSON block in markdown
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire response
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {response_text[:200]}")
            return {}
