"""Asynchronous narrative generation for parallel LLM processing.

Implements the diagram pipeline in parallel:
PGM demographics + OCEAN traits -> LLM A (context) -> LLM B (personas).
"""

import asyncio
import logging
import time
import threading
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

from .models import Persona
from .llm_client import LLMClient, create_llm_client, GenerationConfig
from .exceptions import LLMGenerationError
from .ocean_pipeline import OCEANPipeline
from .narrative_mapping import apply_ocean_to_persona


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a single narrative generation."""
    persona: Persona
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0


class AsyncNarrativeGenerator:
    """
    Parallel narrative generator using thread pools.
    
    Processes multiple LLM calls concurrently to maximize throughput.
    Includes rate limiting and error handling for individual failures.
    
    Example:
        generator = AsyncNarrativeGenerator(
            provider="dashscope",
            model="qwen-mt-flash",
            max_workers=10,
            rate_limit_per_second=5
        )
        enriched = await generator.generate_batch(personas)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: str = "mock",
        config: Optional[GenerationConfig] = None,
        max_workers: int = 10,
        rate_limit_per_second: Optional[float] = None,
        ocean_tolerance: int = 15,
        max_rewrites: int = 3,
        **llm_kwargs
    ):
        """
        Initialize async narrative generator.
        
        Args:
            llm_client: Pre-configured LLM client (optional)
            provider: LLM provider ("openai", "qwen", "dashscope", etc.)
            config: Generation configuration
            max_workers: Maximum concurrent LLM calls
            rate_limit_per_second: Optional rate limit (requests/sec)
            **llm_kwargs: Arguments for LLM client creation
        """
        if llm_client:
            self.client = llm_client
        else:
            self.client = create_llm_client(provider, **llm_kwargs)
        
        self.config = config or GenerationConfig(
            temperature=0.7,
            max_tokens=1200,
        )
        self.max_workers = max_workers
        self.rate_limit = rate_limit_per_second
        self.ocean_tolerance = ocean_tolerance
        self.max_rewrites = max_rewrites
        
        # Rate limiting state
        self._last_request_time = 0
        self._rate_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_latency_ms': 0,
        }

    class _RateLimitedClient(LLMClient):
        """Wrapper to apply rate limiting to LLM calls."""
        def __init__(self, outer: "AsyncNarrativeGenerator"):
            self._outer = outer

        def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
            return self._outer._rate_limited_generate(prompt, config)
    
    def _rate_limited_generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        Generate with optional rate limiting.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Generated text
        """
        if self.rate_limit:
            with self._rate_lock:
                min_interval = 1.0 / self.rate_limit
                elapsed = time.time() - self._last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                self._last_request_time = time.time()
        
        return self.client.generate(prompt, config or self.config)

    def _run_async(self, coro):
        """Run coroutine in current or new event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        if not loop.is_running():
            return loop.run_until_complete(coro)
        result_container = {}

        def runner():
            result_container["value"] = asyncio.run(coro)

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        return result_container.get("value")
    
    def _generate_single_narrative(self, persona: Persona) -> GenerationResult:
        """
        Generate narrative for a single persona.
        
        Args:
            persona: Persona to enrich
            
        Returns:
            GenerationResult with enriched persona or error
        """
        start_time = time.time()
        
        try:
            rate_limited_client = self._RateLimitedClient(self)
            pipeline = OCEANPipeline(
                llm_client=rate_limited_client,
                ocean_tolerance=self.ocean_tolerance,
                max_rewrites=self.max_rewrites,
                generation_config=self.config,
                rewrite_config=self.config,
            )
            profile = self._run_async(pipeline.generate(persona, persona.name))
            persona = apply_ocean_to_persona(persona, profile)
            latency_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                persona=persona,
                success=True,
                latency_ms=latency_ms
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to generate narrative for {persona.uuid}: {e}")
            
            # Mark as failed
            persona.narrative_status = "failed"
            
            return GenerationResult(
                persona=persona,
                success=False,
                error=str(e),
                latency_ms=latency_ms
            )
    
    def generate_batch(
        self,
        personas: List[Persona],
        show_progress: bool = True,
        fail_fast: bool = False
    ) -> List[Persona]:
        """
        Generate narratives for multiple personas in parallel.
        
        Args:
            personas: List of personas to enrich
            show_progress: Whether to show progress bar
            fail_fast: If True, raise on first error; if False, continue
            
        Returns:
            List of enriched personas (failed ones returned as-is)
        """
        self.stats = {
            'total': len(personas),
            'success': 0,
            'failed': 0,
            'total_latency_ms': 0,
        }
        
        results: List[Optional[Persona]] = [None] * len(personas)
        
        # Use ThreadPoolExecutor for concurrent LLM calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with their indices
            future_to_index = {
                executor.submit(self._generate_single_narrative, persona): i
                for i, persona in enumerate(personas)
            }
            
            # Process completed tasks with progress bar
            iterator = as_completed(future_to_index)
            if show_progress:
                iterator = tqdm(
                    iterator, 
                    total=len(personas), 
                    desc="Generating narratives (parallel)"
                )
            
            for future in iterator:
                index = future_to_index[future]
                
                try:
                    result = future.result()
                    results[index] = result.persona
                    
                    self.stats['total_latency_ms'] += result.latency_ms
                    
                    if result.success:
                        self.stats['success'] += 1
                    else:
                        self.stats['failed'] += 1
                        if fail_fast:
                            raise LLMGenerationError(
                                f"Failed to generate for index {index}: {result.error}",
                                provider=getattr(self.client, 'provider', 'unknown')
                            )
                            
                except Exception as e:
                    self.stats['failed'] += 1
                    logger.error(f"Unexpected error for index {index}: {e}")
                    results[index] = personas[index]  # Return original on error
                    
                    if fail_fast:
                        raise
        
        # Log statistics
        if self.stats['success'] > 0:
            avg_latency = self.stats['total_latency_ms'] / self.stats['total']
            logger.info(
                f"Narrative generation complete: {self.stats['success']}/{self.stats['total']} "
                f"succeeded, avg latency: {avg_latency:.0f}ms"
            )
        
        return [p for p in results if p is not None]
    
    def get_stats(self) -> dict:
        """Get generation statistics."""
        stats = self.stats.copy()
        if stats['success'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total']
            stats['success_rate'] = stats['success'] / stats['total']
        return stats
    

def enrich_personas_parallel(
    personas: List[Persona],
    provider: str = "mock",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_workers: int = 10,
    rate_limit_per_second: Optional[float] = None,
    show_progress: bool = True,
    ocean_tolerance: int = 15,
    max_rewrites: int = 3,
) -> List[Persona]:
    """
    Convenience function to enrich personas with parallel narrative generation.
    
    Args:
        personas: List of personas to enrich
        provider: LLM provider ("openai", "qwen", "dashscope", etc.)
        api_key: API key for the provider
        model: Model name
        max_workers: Maximum concurrent LLM calls
        rate_limit_per_second: Optional rate limit
        show_progress: Whether to show progress bar
        
    Returns:
        Enriched personas
        
    Example:
        # Generate 1000 personas with 10 parallel workers
        personas = generator.generate(1000)
        enriched = enrich_personas_parallel(
            personas,
            provider="dashscope",
            model="qwen-mt-flash",
            max_workers=10,
            rate_limit_per_second=5  # 5 requests/sec to avoid throttling
        )
    """
    kwargs = {}
    if api_key:
        kwargs['api_key'] = api_key
    if model:
        kwargs['model'] = model
    
    generator = AsyncNarrativeGenerator(
        provider=provider,
        max_workers=max_workers,
        rate_limit_per_second=rate_limit_per_second,
        ocean_tolerance=ocean_tolerance,
        max_rewrites=max_rewrites,
        **kwargs
    )
    
    return generator.generate_batch(personas, show_progress=show_progress)
