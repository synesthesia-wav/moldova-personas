"""
Narrative generation for personas using LLMs.

Generates the 6 narrative sections for each persona and populates
structured skill/hobby lists.
"""

import logging
from typing import List, Optional, Callable, Dict, Tuple
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


class NarrativeGenerator:
    """
    Generates narrative content for personas.
    """
    
    def __init__(self, 
                 llm_client: Optional[LLMClient] = None,
                 provider: str = "mock",
                 config: Optional[GenerationConfig] = None,
                 allow_mock_narratives: bool = True,
                 **llm_kwargs):
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
            max_tokens=600,  # Increased for 6 sections
        )
        self.allow_mock_narratives = allow_mock_narratives
    
    def generate_for_persona(self, persona: Persona) -> Persona:
        """
        Generate narrative content for a single persona.
        
        Args:
            persona: Persona with structured fields
        
        Returns:
            Persona with narrative fields populated
        """
        required_fields, min_lengths = _get_required_fields_and_lengths()
        prompt = generate_full_prompt(
            persona,
            required_keys=required_fields,
            min_lengths=min_lengths,
        )
        
        # Generate with LLM
        try:
            response = self.client.generate(prompt, self.config)
        except LLMGenerationError as e:
            logger.error(f"LLM generation failed for persona {persona.uuid}: {e}")
            # Mark as failed
            persona.narrative_status = "failed"
            return persona
        
        # Check if response is empty (mock mode)
        if not response or not response.strip():
            if self.allow_mock_narratives:
                persona.narrative_status = "mock"
                return persona
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
                repair_response = self.client.generate(repair_prompt, self.config)
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
                repair_response = self.client.generate(repair_prompt, self.config)
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
        # Empty string indicates no LLM-generated content available
        if not persona.cultural_background:
            persona.cultural_background = ""
        
        # Mark as successfully generated
        if persona.narrative_status != "failed":
            persona.narrative_status = "generated"
        
        return persona
    
    def generate_batch(self, 
                       personas: List[Persona],
                       show_progress: bool = True,
                       delay: float = 0.0) -> List[Persona]:
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
    allow_mock_narratives: bool = True,
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
        kwargs['api_key'] = api_key
    if model:
        kwargs['model'] = model
    
    generator = NarrativeGenerator(
        provider=provider,
        allow_mock_narratives=allow_mock_narratives,
        **kwargs,
    )
    return generator.generate_batch(personas, delay=delay)
