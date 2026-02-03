"""Asynchronous narrative generation for parallel LLM processing.

Implements concurrent LLM API calls using ThreadPoolExecutor
to significantly speed up narrative generation for large batches.
"""

import logging
import time
import threading
from typing import List, Optional, Callable, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

from .models import Persona
from .llm_client import LLMClient, create_llm_client, GenerationConfig
from .exceptions import LLMGenerationError
from .prompts import (
    generate_full_prompt,
    parse_json_narrative_response_strict,
    extract_skills_from_text,
    extract_hobbies_from_text,
    generate_repair_prompt,
)
from .narrative_contract import NARRATIVE_JSON_SCHEMA


logger = logging.getLogger(__name__)


def _get_required_fields_and_lengths() -> Tuple[List[str], Dict[str, int]]:
    """Load required narrative keys and min lengths from JSON schema."""
    required_fields = NARRATIVE_JSON_SCHEMA.get("required", [])
    properties = NARRATIVE_JSON_SCHEMA.get("properties", {})
    min_lengths = {
        field: properties.get(field, {}).get("minLength", 1)
        for field in required_fields
    }
    return required_fields, min_lengths


def _missing_fields(
    narratives: Dict[str, str],
    required_fields: List[str],
    min_lengths: Dict[str, int],
) -> List[str]:
    """Return fields that are missing or shorter than required min length."""
    missing = []
    for field in required_fields:
        value = narratives.get(field, "")
        min_len = min_lengths.get(field, 1)
        if not isinstance(value, str) or len(value.strip()) < min_len:
            missing.append(field)
    return missing


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
        allow_mock_narratives: bool = True,
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
            max_tokens=600,
        )
        self.max_workers = max_workers
        self.rate_limit = rate_limit_per_second
        self.allow_mock_narratives = allow_mock_narratives
        
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
    
    def _rate_limited_generate(self, prompt: str) -> str:
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
        
        return self.client.generate(prompt, self.config)
    
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
            required_fields, min_lengths = _get_required_fields_and_lengths()
            prompt = generate_full_prompt(
                persona,
                required_keys=required_fields,
                min_lengths=min_lengths,
            )
            
            # Generate with LLM (rate limited)
            response = self._rate_limited_generate(prompt)
            
            # Check if response is empty (mock mode)
            if not response or not response.strip():
                if self.allow_mock_narratives:
                    persona.narrative_status = "mock"
                    latency_ms = (time.time() - start_time) * 1000
                    
                    return GenerationResult(
                        persona=persona,
                        success=True,  # Not a failure, just mock mode
                        latency_ms=latency_ms
                    )
                response = ""
            
            # Parse response (strict JSON-only)
            try:
                narratives = parse_json_narrative_response_strict(
                    response,
                    required_fields,
                    require_all=True,
                )
            except Exception as e:
                logger.warning(f"Strict JSON parse failed for {persona.uuid}: {e}")
                narratives = {}

            missing_fields = _missing_fields(narratives, required_fields, min_lengths)
            if missing_fields:
                try:
                    repair_prompt = generate_repair_prompt(
                        persona,
                        missing_fields,
                        existing_sections=narratives,
                        min_lengths=min_lengths,
                        force_fill_all=False,
                    )
                    repair_response = self._rate_limited_generate(repair_prompt)
                    if repair_response and repair_response.strip():
                        try:
                            repaired = parse_json_narrative_response_strict(
                                repair_response,
                                missing_fields,
                                require_all=False,
                            )
                        except Exception as e:
                            logger.warning(f"Repair parse failed for {persona.uuid}: {e}")
                            repaired = {}
                        for field in missing_fields:
                            candidate = repaired.get(field, "")
                            min_len = min_lengths.get(field, 1)
                            if isinstance(candidate, str) and len(candidate.strip()) >= min_len:
                                narratives[field] = candidate.strip()
                except Exception as e:
                    logger.warning(f"Repair prompt failed for {persona.uuid}: {e}")

            missing_fields = _missing_fields(narratives, required_fields, min_lengths)
            if missing_fields:
                try:
                    repair_prompt = generate_repair_prompt(
                        persona,
                        missing_fields,
                        existing_sections=narratives,
                        min_lengths=min_lengths,
                        force_fill_all=True,
                    )
                    repair_response = self._rate_limited_generate(repair_prompt)
                    if repair_response and repair_response.strip():
                        try:
                            repaired = parse_json_narrative_response_strict(
                                repair_response,
                                missing_fields,
                                require_all=False,
                            )
                        except Exception as e:
                            logger.warning(f"Repair-2 parse failed for {persona.uuid}: {e}")
                            repaired = {}
                        for field in missing_fields:
                            candidate = repaired.get(field, "")
                            min_len = min_lengths.get(field, 1)
                            if isinstance(candidate, str) and len(candidate.strip()) >= min_len:
                                narratives[field] = candidate.strip()
                except Exception as e:
                    logger.warning(f"Repair-2 prompt failed for {persona.uuid}: {e}")

            missing_fields = _missing_fields(narratives, required_fields, min_lengths)
            if missing_fields:
                persona.narrative_status = "failed"
            
            # Update persona
            persona.descriere_generala = narratives.get("descriere_generala", "")
            persona.profil_profesional = narratives.get("profil_profesional", "")
            persona.hobby_sport = narratives.get("hobby_sport", "")
            persona.hobby_arta_cultura = narratives.get("hobby_arta_cultura", "")
            persona.hobby_calatorii = narratives.get("hobby_calatorii", "")
            persona.hobby_culinar = narratives.get("hobby_culinar", "")
            persona.career_goals_and_ambitions = narratives.get("career_goals_and_ambitions", "")
            persona.persona_summary = narratives.get("persona_summary", "")
            
            # Extract structured lists
            all_hobby_text = " ".join([
                persona.hobby_sport,
                persona.hobby_arta_cultura,
                persona.hobby_calatorii,
                persona.hobby_culinar
            ])
            
            persona.skills_and_expertise_list = extract_skills_from_text(
                persona.profil_profesional
            )
            persona.hobbies_and_interests_list = extract_hobbies_from_text(all_hobby_text)
            
            # Cultural background is generated by LLM only - no template fallbacks
            if not persona.cultural_background:
                persona.cultural_background = ""
            
            # Mark as successfully generated
            if persona.narrative_status != "failed":
                persona.narrative_status = "generated"
            
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
    allow_mock_narratives: bool = True,
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
        allow_mock_narratives=allow_mock_narratives,
        **kwargs
    )
    
    return generator.generate_batch(personas, show_progress=show_progress)
