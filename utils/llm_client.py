"""
LLM API Client for Anthropic and OpenAI
Provides unified interface for multiple LLM providers.
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import os


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2000):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0
        self.total_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            'model': self.model,
            'call_count': self.call_count,
            'total_tokens': self.total_tokens
        }


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        # Initialize client
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key.")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from Claude."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Update stats
        self.call_count += 1
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens

        return response.content[0].text


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Initialize client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from GPT."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Update stats
        self.call_count += 1
        self.total_tokens += response.usage.total_tokens

        return response.choices[0].message.content


def get_llm_client(provider: str, model: str, api_key: Optional[str] = None,
                   **kwargs) -> BaseLLMClient:
    """
    Factory function to get LLM client.

    Args:
        provider: "anthropic" or "openai"
        model: Model name
        api_key: API key (optional, can use env var)
        **kwargs: Additional arguments (temperature, max_tokens, etc.)

    Returns:
        LLM client instance
    """
    if provider.lower() == "anthropic":
        return AnthropicClient(model=model, api_key=api_key, **kwargs)
    elif provider.lower() == "openai":
        return OpenAIClient(model=model, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")
