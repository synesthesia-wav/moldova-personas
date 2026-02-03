"""
Narrative generation for personas using LLMs.

Implements the diagram pipeline:
PGM demographics + OCEAN traits -> LLM A (context fields) -> LLM B (persona variants).
"""

import asyncio
import logging
import threading
from typing import List, Optional

from tqdm import tqdm

from .models import Persona
from .llm_client import LLMClient, create_llm_client, GenerationConfig
from .ocean_pipeline import OCEANPipeline
from .narrative_mapping import apply_ocean_to_persona


logger = logging.getLogger(__name__)


class NarrativeGenerator:
    """
    Generates narrative content for personas.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: str = "mock",
        config: Optional[GenerationConfig] = None,
        ocean_tolerance: int = 15,
        max_rewrites: int = 3,
        **llm_kwargs,
    ):
        """
        Initialize narrative generator.

        Args:
            llm_client: Pre-configured LLM client (optional)
            provider: LLM provider ("openai", "local", "mock")
            config: Generation configuration
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
        self.pipeline = OCEANPipeline(
            llm_client=self.client,
            ocean_tolerance=ocean_tolerance,
            max_rewrites=max_rewrites,
            generation_config=self.config,
            rewrite_config=self.config,
        )

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

    def generate_for_persona(self, persona: Persona) -> Persona:
        """
        Generate narrative content for a single persona.

        Args:
            persona: Persona with structured fields

        Returns:
            Persona with narrative fields populated
        """
        try:
            profile = self._run_async(self.pipeline.generate(persona, persona.name))
        except Exception as e:
            logger.error(f"LLM generation failed for persona {persona.uuid}: {e}")
            persona.narrative_status = "failed"
            return persona

        return apply_ocean_to_persona(persona, profile)

    def generate_batch(
        self,
        personas: List[Persona],
        show_progress: bool = True,
        delay: float = 0.0,
    ) -> List[Persona]:
        """
        Generate narratives for multiple personas.

        Args:
            personas: List of personas
            show_progress: Whether to show progress bar
            delay: Delay between API calls (to avoid rate limits)

        Returns:
            List of personas with narratives
        """
        iterator = enumerate(personas)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Generating narratives")

        results = []
        for i, persona in iterator:
            try:
                enriched = self.generate_for_persona(persona)
                results.append(enriched)

                if delay > 0 and i < len(personas) - 1:
                    import time

                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Error generating for {persona.uuid}: {e}")
                results.append(persona)  # Return original on error

        return results



def enrich_personas_with_narratives(
    personas: List[Persona],
    provider: str = "mock",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    delay: float = 0.0,
    ocean_tolerance: int = 15,
    max_rewrites: int = 3,
) -> List[Persona]:
    """
    Convenience function to enrich personas with narratives.

    Args:
        personas: List of personas to enrich
        provider: LLM provider ("openai", "local", "mock")
        api_key: API key for OpenAI
        model: Model name
        delay: Delay between API calls

    Returns:
        Enriched personas
    """
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if model:
        kwargs["model"] = model

    generator = NarrativeGenerator(
        provider=provider,
        ocean_tolerance=ocean_tolerance,
        max_rewrites=max_rewrites,
        **kwargs,
    )
    return generator.generate_batch(personas, delay=delay)
