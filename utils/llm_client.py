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


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, model: str = "gemini-2.0-flash",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        # Initialize client
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key.")

        self.genai.configure(api_key=api_key)
        self.client = self.genai.GenerativeModel(model)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from Gemini."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # Configure generation
        generation_config = self.genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Update stats
        self.call_count += 1
        # Gemini provides token counts in usage_metadata
        if hasattr(response, 'usage_metadata'):
            self.total_tokens += (
                response.usage_metadata.prompt_token_count +
                response.usage_metadata.candidates_token_count
            )

        return response.text


class HuggingFaceClient(BaseLLMClient):
    """Client for HuggingFace local models."""

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 device: str = "auto"):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.torch = torch
            self.pipeline = pipeline
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. "
                "Run: pip install transformers torch accelerate"
            )

        print(f"Loading HuggingFace model: {model}")
        print("This may take a few minutes on first load...")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )

        # Create pipeline
        self.pipe = self.pipeline(
            "text-generation",
            model=self.model_obj,
            tokenizer=self.tokenizer,
            device_map=device
        )

        print(f"Model loaded successfully on device: {device}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from HuggingFace model."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # For instruct models, format the prompt
        if "instruct" in self.model.lower() or "chat" in self.model.lower():
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Generate
        outputs = self.pipe(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Extract generated text
        generated_text = outputs[0]["generated_text"]

        # Remove the prompt from output (for non-instruct models)
        if not ("instruct" in self.model.lower() or "chat" in self.model.lower()):
            generated_text = generated_text[len(formatted_prompt):].strip()
        else:
            # For instruct models, extract only the assistant response
            if formatted_prompt in generated_text:
                generated_text = generated_text[len(formatted_prompt):].strip()

        # Update stats
        self.call_count += 1
        # Approximate token count
        self.total_tokens += len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(generated_text))

        return generated_text


def get_llm_client(provider: str, model: str, api_key: Optional[str] = None,
                   **kwargs) -> BaseLLMClient:
    """
    Factory function to get LLM client.

    Args:
        provider: "anthropic", "openai", "gemini", or "huggingface"
        model: Model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022",
              "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-pro-preview",
              "meta-llama/Llama-3.1-8B-Instruct")
        api_key: API key (optional, can use env var)
        **kwargs: Additional arguments (temperature, max_tokens, device, etc.)

    Returns:
        LLM client instance
    """
    provider = provider.lower()

    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model=model, api_key=api_key, **kwargs)
    elif provider == "gemini":
        return GeminiClient(model=model, api_key=api_key, **kwargs)
    elif provider == "huggingface":
        return HuggingFaceClient(model=model, api_key=api_key, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Use 'anthropic', 'openai', 'gemini', or 'huggingface'."
        )
