"""
LLM client for generating narrative content.

Supports:
- OpenAI API (GPT-3.5, GPT-4)
- Local models via transformers
- Mock mode for testing
"""

import os
import time
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .exceptions import LLMGenerationError

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False


logger = logging.getLogger(__name__)


def with_retry(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator that adds retry logic with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except LLMGenerationError as e:
                    if not e.retryable or attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
            return None  # Should never reach here
        return wrapper
    return decorator


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 300
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text from a single prompt."""
        pass
    
    def generate_batch(self, prompts: List[str], 
                       config: Optional[GenerationConfig] = None,
                       delay: float = 0.0) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, config))
            if delay > 0:
                time.sleep(delay)
        return results


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
    
    @with_retry(max_retries=3, backoff_factor=2.0)
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ești un asistent care generează descrieri realiste de personae în limba română. Folosește un ton natural și autentic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )
            return response.choices[0].message.content.strip()
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMGenerationError(
                f"OpenAI API error: {e}", 
                provider="openai",
                retryable=e.status_code >= 500 or e.status_code == 429
            ) from e
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            raise LLMGenerationError(
                f"Rate limit exceeded: {e}",
                provider="openai",
                retryable=True
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI generation: {e}")
            raise LLMGenerationError(
                f"Unexpected error: {e}",
                provider="openai",
                retryable=False
            ) from e


class LocalModelClient(LLMClient):
    """Client for local HuggingFace models."""
    
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta", device: str = "auto"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.device = device
        self.pipe = None
        # Lazy loading - don't initialize until first use
        # self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize the text generation pipeline."""
        logger.info(f"Loading model {self.model_name}...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            device=self.device,
            torch_dtype="auto",
        )
        logger.info("Model loaded.")
    
    @with_retry(max_retries=2, backoff_factor=1.5)
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        
        # Lazy initialization
        if self.pipe is None:
            self._init_pipeline()
        
        messages = [
            {"role": "system", "content": "Ești un asistent care generează descrieri realiste de personae în limba română."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            output = self.pipe(
                messages,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=True,
                return_full_text=False,
            )
            return output[0]["generated_text"].strip()
        except ImportError as e:
            logger.error(f"Transformers import error: {e}")
            raise LLMGenerationError(
                f"Transformers not properly installed: {e}",
                provider="local",
                retryable=False
            ) from e
        except RuntimeError as e:
            logger.error(f"Runtime error in local model generation: {e}")
            raise LLMGenerationError(
                f"Model runtime error: {e}",
                provider="local",
                retryable="out of memory" in str(e).lower()
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in local model generation: {e}")
            raise LLMGenerationError(
                f"Unexpected error: {e}",
                provider="local",
                retryable=False
            ) from e


class QwenLocalClient(LLMClient):
    """Client for local Qwen models via HuggingFace."""
    
    DEFAULT_MODELS = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
        "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    }
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 device: str = "auto",
                 load_in_4bit: bool = False):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers package required. Install with: "
                "pip install transformers torch accelerate"
            )
        
        # Resolve shorthand names
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]
        
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.pipe = None
        self.tokenizer = None
        self.model = None
        # Delay initialization until first use to allow for lazy loading
        # self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize the Qwen pipeline."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError as e:
            raise ImportError(
                f"transformers and torch required for local Qwen. "
                f"Install with: pip install transformers torch accelerate"
            ) from e
        
        logger.info(f"Loading Qwen model {self.model_name}...")
        logger.info("(First run will download model weights - this may take several minutes)")
        
        # Configure quantization for memory efficiency
        quantization_config = None
        if self.load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                logger.warning("bitsandbytes not installed. Install with: pip install bitsandbytes")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        # Handle device mapping
        if self.device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device
        
        # Handle dtype
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        logger.info("Qwen model loaded")
    
    @with_retry(max_retries=2, backoff_factor=1.5)
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        
        # Lazy initialization
        if self.model is None:
            self._init_pipeline()
        
        # Format for Qwen chat
        messages = [
            {"role": "system", "content": "Ești un asistent care generează descrieri realiste de personae în limba română. Folosește un ton natural și autentic."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except ImportError as e:
            logger.error(f"Transformers/torch import error: {e}")
            raise LLMGenerationError(
                f"Required packages not installed: {e}",
                provider="qwen-local",
                retryable=False
            ) from e
        except RuntimeError as e:
            logger.error(f"Runtime error in Qwen local generation: {e}")
            raise LLMGenerationError(
                f"Model runtime error (possibly OOM): {e}",
                provider="qwen-local",
                retryable="out of memory" in str(e).lower() or "cuda" in str(e).lower()
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in Qwen local generation: {e}")
            raise LLMGenerationError(
                f"Unexpected error: {e}",
                provider="qwen-local",
                retryable=False
            ) from e


class DashScopeClient(LLMClient):
    """Client for Alibaba DashScope API (Qwen models)."""
    
    MODEL_MAPPING = {
        "qwen-turbo": "qwen-turbo",
        "qwen-plus": "qwen-plus",
        "qwen-max": "qwen-max",
        "qwen2.5-7b": "qwen2.5-7b-instruct",
        "qwen2.5-14b": "qwen2.5-14b-instruct",
        "qwen2.5-72b": "qwen2.5-72b-instruct",
        "qwen-mt-flash": "qwen-mt-flash",  # Multilingual translation model
        "qwen-mt-turbo": "qwen-mt-turbo",  # Multilingual translation model
    }
    
    def __init__(self, model: str = "qwen-turbo", api_key: Optional[str] = None, 
                 base_url: Optional[str] = None):
        if not HAS_DASHSCOPE:
            raise ImportError(
                "dashscope package required. Install with: pip install dashscope"
            )
        
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key required. Set DASHSCOPE_API_KEY env var."
            )
        
        dashscope.api_key = self.api_key
        
        # Use international endpoint by default (works with most API keys)
        # Can override with DASHSCOPE_BASE_URL env var or base_url parameter
        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1")
        dashscope.base_http_api_url = self.base_url
        
        # Resolve model name
        self.model = self.MODEL_MAPPING.get(model, model)
    
    # Models that don't support system messages
    NO_SYSTEM_MODELS = {"qwen-mt-flash", "qwen-mt-turbo"}
    
    @with_retry(max_retries=3, backoff_factor=2.0)
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        
        from dashscope import Generation
        
        # Some models don't support system messages, so we adapt
        system_content = "Ești un asistent care generează descrieri realiste de personae în limba română. Folosește un ton natural și autentic, cu diacritice corecte."
        
        if self.model in self.NO_SYSTEM_MODELS:
            # Combine system prompt with user prompt for models without system support
            combined_prompt = f"{system_content}\n\n{prompt}"
            messages = [
                {"role": "user", "content": combined_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        
        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                result_format="message"
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content.strip()
            else:
                error_msg = f"DashScope API error: {response.message} (code: {response.status_code})"
                logger.error(error_msg)
                raise LLMGenerationError(
                    error_msg,
                    provider="dashscope",
                    retryable=response.status_code >= 500 or response.status_code == 429
                )
                
        except LLMGenerationError:
            raise  # Re-raise our own exceptions
        except ImportError as e:
            logger.error(f"DashScope import error: {e}")
            raise LLMGenerationError(
                f"DashScope not installed: {e}",
                provider="dashscope",
                retryable=False
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in DashScope generation: {e}")
            raise LLMGenerationError(
                f"Unexpected error: {e}",
                provider="dashscope",
                retryable=False
            ) from e


class MockLLMClient(LLMClient):
    """
    Mock client for testing without API calls.
    
    WARNING: This client returns EMPTY content. It is for testing infrastructure
    only and should NEVER be used for production data generation. Using mock
    mode will result in all narrative fields being empty strings.
    
    To generate actual narrative content, use a real LLM provider:
    - "dashscope" (Qwen via Alibaba Cloud)
    - "openai" (GPT models)
    - "qwen-local" (local Qwen models)
    """
    
    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self._warned = False
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        if not self._warned:
            logger.warning(
                "MOCK MODE ACTIVE: LLM is not configured. "
                "All narrative fields will be EMPTY. "
                "Use a real provider (dashscope, openai, etc.) for actual content generation."
            )
            self._warned = True
        time.sleep(self.delay)
        # Return empty string - no fabricated content
        return ""


def create_llm_client(provider: str = "mock", **kwargs) -> LLMClient:
    """
    Factory function to create appropriate LLM client.
    
    Args:
        provider: One of "openai", "qwen", "qwen-local", "local", "dashscope", "mock"
        **kwargs: Provider-specific arguments
    
    Returns:
        LLMClient instance
    
    Examples:
        # OpenAI
        client = create_llm_client("openai", model="gpt-3.5-turbo")
        
        # Qwen via DashScope API
        client = create_llm_client("qwen", model="qwen-turbo")
        
        # Qwen local (requires ~16GB RAM for 7B, ~28GB for 14B)
        client = create_llm_client("qwen-local", model_name="qwen2.5-7b")
        
        # Qwen local with 4-bit quantization (saves VRAM)
        client = create_llm_client("qwen-local", model_name="qwen2.5-14b", load_in_4bit=True)
    """
    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider in ("qwen", "dashscope"):
        return DashScopeClient(**kwargs)
    elif provider == "qwen-local":
        return QwenLocalClient(**kwargs)
    elif provider == "local":
        return LocalModelClient(**kwargs)
    elif provider == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
