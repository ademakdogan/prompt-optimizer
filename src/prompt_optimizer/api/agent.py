"""
Agent model handler for processing data with prompts.

The agent model processes input data using a given prompt and
returns structured PII extraction results.
"""

from prompt_optimizer.api.client import OpenRouterClient
from prompt_optimizer.config import get_settings
from prompt_optimizer.models import PIIResponse
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class AgentModel:
    """
    Agent model for processing data with prompts.

    The agent takes a prompt and input data, then returns
    structured PII extraction results.

    Attributes:
        client: The OpenRouter client instance.

    Examples:
        >>> agent = AgentModel()
        >>> response = agent.process_data(
        ...     prompt="Extract PII entities from the text.",
        ...     data="Contact John at john@email.com"
        ... )
        >>> isinstance(response, PIIResponse)
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
        logger.info(f"Initialized AgentModel with model: {model_name}")

    def process_data(
        self,
        prompt: str,
        data: str,
        schema_description: str = "",
    ) -> PIIResponse:
        """
        Process input data using the given prompt.

        Args:
            prompt: The instruction prompt for PII extraction.
            data: The input text to analyze.
            schema_description: Optional schema description for the model.

        Returns:
            PIIResponse: Structured response with extracted PII entities.

        Examples:
            >>> agent = AgentModel()
            >>> result = agent.process_data(
            ...     prompt="Find all email addresses",
            ...     data="Email: test@example.com"
            ... )
            >>> len(result.entities) >= 0
            True
        """
        logger.debug(f"Processing data with prompt: {prompt[:50]}...")
        
        system_message = self._build_system_message(prompt, schema_description)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Analyze this text:\n\n{data}"},
        ]
        
        response = self.client.chat(
            messages=messages,
            response_model=PIIResponse,
            temperature=0.3,  # Lower temperature for more consistent extraction
        )
        
        logger.debug(f"Extracted {len(response.entities)} entities")
        return response

    def _build_system_message(self, prompt: str, schema_description: str) -> str:
        """
        Build the system message for the API request.

        Args:
            prompt: The user's extraction prompt.
            schema_description: Optional schema description.

        Returns:
            str: The formatted system message.
        """
        base_message = f"""You are a PII (Personally Identifiable Information) extraction assistant.

{prompt}

Your task is to:
1. Identify all PII entities in the given text
2. For each entity, provide its value, label (type), and position (start/end)
3. Create a masked version of the text with PII replaced by labels in brackets

Common PII types include: FIRSTNAME, LASTNAME, EMAIL, PHONE, ADDRESS, SSN, 
CREDITCARDNUMBER, DATE, TIME, URL, IP, USERNAME, PASSWORD, etc.

Be thorough and accurate in your extraction."""

        if schema_description:
            base_message += f"\n\nExpected output schema:\n{schema_description}"
        
        return base_message
