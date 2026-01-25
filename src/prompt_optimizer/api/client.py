"""
OpenRouter API client for Prompt Optimizer.

This module provides a client for interacting with the OpenRouter API,
supporting structured output via instructor library.
"""

from typing import Any, Type, TypeVar

import httpx
import instructor
from openai import OpenAI
from pydantic import BaseModel

from prompt_optimizer.config import get_settings
from prompt_optimizer.utils.logging import get_logger

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)


class OpenRouterClient:
    """
    Client for interacting with OpenRouter API.

    This client wraps the OpenAI SDK configured for OpenRouter,
    with instructor patching for structured output.

    Attributes:
        client: The instructor-patched OpenAI client.
        model: The model identifier to use.

    Examples:
        >>> client = OpenRouterClient(model="openai/gpt-5-nano")
        >>> # Use client.chat() for structured responses
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """
        Initialize the OpenRouter client.

        Args:
            model: The model identifier (e.g., "openai/gpt-5-nano").
            api_key: OpenRouter API key. If None, loaded from settings.

        Raises:
            ValueError: If no API key is provided or found in settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.openrouter_api_key

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY in .env or pass api_key parameter."
            )

        self.model = model
        
        # Create OpenAI client configured for OpenRouter
        base_client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/prompt-optimizer",
                "X-Title": "Prompt Optimizer",
            },
        )
        
        # Patch with instructor for structured output
        self.client = instructor.from_openai(base_client)

        logger.debug(f"Initialized OpenRouter client with model: {model}")

    def chat(
        self,
        messages: list[dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_retries: int = 3,
        reasoning_effort: str = "low",
    ) -> T:
        """
        Send a chat completion request with structured output.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_model: Pydantic model class for response parsing.
            temperature: Sampling temperature (0.0 to 2.0).
            max_retries: Number of retries on failure.
            reasoning_effort: Reasoning effort level ("low", "medium", "high").
                             Use "low" to minimize thinking tokens.

        Returns:
            An instance of response_model populated with the response.

        Raises:
            Exception: If API call fails after all retries.

        Examples:
            >>> from prompt_optimizer.models import TargetResult
            >>> messages = [{"role": "user", "content": "Extract PII from: John"}]
            >>> response = client.chat(messages, TargetResult)
            >>> isinstance(response, TargetResult)
            True
        """
        logger.debug(f"Sending chat request to {self.model}")
        
        try:
            # Build extra parameters for OpenRouter/OpenAI
            extra_body = {}
            
            # Add reasoning_effort for models that support it (like gpt-5)
            if reasoning_effort and "gpt-5" in self.model:
                extra_body["reasoning_effort"] = reasoning_effort
                logger.debug(f"Using reasoning_effort: {reasoning_effort}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_retries=max_retries,
                extra_body=extra_body if extra_body else None,
            )
            logger.debug("Received structured response from API")
            return response
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    def chat_raw(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        reasoning_effort: str = "low",
    ) -> str:
        """
        Send a chat completion request and return raw text response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0.0 to 2.0).
            reasoning_effort: Reasoning effort level ("low", "medium", "high").

        Returns:
            The raw text content from the model response.

        Examples:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = client.chat_raw(messages)
            >>> isinstance(response, str)
            True
        """
        logger.debug(f"Sending raw chat request to {self.model}")
        
        # Use the underlying OpenAI client directly for raw response
        base_client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/prompt-optimizer",
                "X-Title": "Prompt Optimizer",
            },
        )
        
        # Build extra parameters
        extra_body = {}
        if reasoning_effort and "gpt-5" in self.model:
            extra_body["reasoning_effort"] = reasoning_effort
        
        response = base_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body if extra_body else None,
        )
        
        content = response.choices[0].message.content or ""
        logger.debug(f"Received raw response: {content[:100]}...")
        return content
