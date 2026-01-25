"""
Agent model handler for processing data with prompts.

The agent model processes input data using a given prompt and
returns structured PII extraction results using TargetResult model.
"""

from prompt_optimizer.api.client import OpenRouterClient
from prompt_optimizer.config import get_settings
from prompt_optimizer.models import TargetResult
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class AgentModel:
    """
    Agent model for processing data with prompts.

    The agent takes a prompt and input data, then returns
    structured PII extraction results as TargetResult.

    Attributes:
        client: The OpenRouter client instance.
        field_descriptions: Current field descriptions for schema.

    Examples:
        >>> agent = AgentModel()
        >>> response = agent.process_data(
        ...     prompt="Extract PII entities from the text.",
        ...     data="Contact John at john@email.com"
        ... )
        >>> isinstance(response, TargetResult)
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
            >>> agent.field_descriptions["email"]
            'Email in user@domain format'
        """
        self.field_descriptions.update(descriptions)
        logger.info(f"Updated field descriptions: {list(descriptions.keys())}")

    def process_data(
        self,
        prompt: str,
        data: str,
        schema_description: str = "",
    ) -> TargetResult:
        """
        Process input data using the given prompt.

        Args:
            prompt: The instruction prompt for PII extraction.
            data: The input text to analyze.
            schema_description: Optional schema description for the model.

        Returns:
            TargetResult: Structured response with extracted PII fields.

        Examples:
            >>> agent = AgentModel()
            >>> result = agent.process_data(
            ...     prompt="Find all email addresses",
            ...     data="Email: test@example.com"
            ... )
            >>> hasattr(result, 'email')
            True
        """
        logger.debug(f"Processing data with prompt: {prompt[:50]}...")
        
        system_message = self._build_system_message(prompt, schema_description)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Analyze this text and extract all PII:\n\n{data}"},
        ]
        
        response = self.client.chat(
            messages=messages,
            response_model=TargetResult,
            temperature=0.3,  # Lower temperature for more consistent extraction
        )
        
        # Count non-None fields
        extracted_count = sum(
            1 for k, v in response.model_dump().items() if v is not None
        )
        logger.debug(f"Extracted {extracted_count} PII fields")
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
        # Build field descriptions section
        field_desc_section = ""
        if self.field_descriptions:
            lines = [f"- {name}: {desc}" for name, desc in self.field_descriptions.items()]
            field_desc_section = f"""

FIELD DESCRIPTIONS (use these to understand what to extract):
{chr(10).join(lines)}
"""

        base_message = f"""You are a PII (Personally Identifiable Information) extraction assistant.

{prompt}

Your task is to extract PII from the given text and return a JSON object
with the following field names (only include fields that are present in the text):

- firstname: First name of a person
- lastname: Last name/surname of a person
- prefix: Title prefix like Mr., Mrs., Dr.
- email: Email address
- phonenumber: Phone number
- age: Age of a person
- street: Street address
- city: City name
- county: County or region name
- country: Country name
- zipcode: Postal/ZIP code
- username: Username or user ID
- password: Password
- pin: PIN code
- accountnumber: Bank or account number
- maskednumber: Masked credit card number (last 4 digits)
- time: Time value
- date: Date value
- amount: Monetary amount
- currency: Currency code (USD, EUR, etc.)
- jobtitle: Job title or position
- eyecolor: Eye color description
- nearbygpscoordinate: GPS coordinates
- useragent: Browser user agent string
- ipaddress: IP address
- url: URL or web address
{field_desc_section}
Be thorough and accurate in your extraction. Only include fields that are found in the text."""

        if schema_description:
            base_message += f"\n\nAdditional schema notes:\n{schema_description}"
        
        return base_message
