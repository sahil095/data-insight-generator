"""Unified LLM client supporting Groq and OpenAI APIs."""
from typing import Optional, Dict, Any, List
from enum import Enum

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config.settings import settings


class LLMProvider(str, Enum):
    """LLM provider options."""
    GROQ = "groq"
    OPENAI = "openai"


class UnifiedLLMClient:
    """Unified LLM client that supports both Groq and OpenAI APIs."""
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: 'groq' or 'openai'. If None, uses settings.llm_provider
        """
        self.provider = (provider or settings.llm_provider).lower()
        
        if self.provider == LLMProvider.GROQ.value:
            if not GROQ_AVAILABLE:
                raise ImportError("Groq package not installed. Install with: pip install groq")
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is required for Groq provider")
            self.client = Groq(api_key=settings.groq_api_key)
            self.base_url = None
        elif self.provider == LLMProvider.OPENAI.value:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.base_url = None
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def chat_completions_create(
        self,
        model: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Create a chat completion.
        
        Args:
            model: Model name (overrides default from settings)
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            Response object with .choices[0].message.content
        """
        if model is None:
            model = settings.llm_model
        
        if temperature is None:
            temperature = settings.llm_temperature
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Groq uses 'response_format', OpenAI also supports it
        if response_format:
            params["response_format"] = response_format
        
        # Both APIs have similar interfaces
        response = self.client.chat.completions.create(**params)
        
        return response
    
    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        if self.provider == LLMProvider.GROQ.value:
            return GROQ_AVAILABLE and settings.groq_api_key is not None
        elif self.provider == LLMProvider.OPENAI.value:
            return OPENAI_AVAILABLE and settings.openai_api_key is not None
        return False

