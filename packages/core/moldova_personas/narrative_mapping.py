"""
Helpers for mapping OCEAN pipeline outputs into Persona objects.
"""

from __future__ import annotations

import ast
from typing import List, Any

from .models import Persona
from .ocean_pipeline import OCEANPersona
from .prompts import extract_skills_from_text, extract_hobbies_from_text


def _parse_list(value: Any) -> List[str]:
    """Parse list-like fields robustly."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                return []
        # Fallback: single string entry
        return [text]
    return []


def apply_ocean_to_persona(base: Persona, profile: OCEANPersona) -> Persona:
    """Populate Persona fields from an OCEANPersona result."""
    base.persona = profile.persona
    base.professional_persona = profile.professional_persona
    base.sports_persona = profile.sports_persona
    base.arts_persona = profile.arts_persona
    base.travel_persona = profile.travel_persona
    base.culinary_persona = profile.culinary_persona

    base.cultural_background = profile.cultural_background
    base.skills_and_expertise = profile.skills_and_expertise
    base.hobbies_and_interests = profile.hobbies_and_interests
    base.career_goals_and_ambitions = profile.career_goals_and_ambitions

    skills_list = _parse_list(profile.skills_and_expertise_list)
    hobbies_list = _parse_list(profile.hobbies_and_interests_list)

    if not skills_list and base.skills_and_expertise:
        skills_list = extract_skills_from_text(base.skills_and_expertise)
    if not hobbies_list and base.hobbies_and_interests:
        hobbies_list = extract_hobbies_from_text(base.hobbies_and_interests)

    base.skills_and_expertise_list = skills_list
    base.hobbies_and_interests_list = hobbies_list

    # Backward-compatible narrative fields
    base.descriere_generala = profile.persona
    base.profil_profesional = profile.professional_persona
    base.hobby_sport = profile.sports_persona
    base.hobby_arta_cultura = profile.arts_persona
    base.hobby_calatorii = profile.travel_persona
    base.hobby_culinar = profile.culinary_persona
    base.persona_summary = profile.persona

    # OCEAN traits
    base.ocean_openness = profile.ocean_openness
    base.ocean_conscientiousness = profile.ocean_conscientiousness
    base.ocean_extraversion = profile.ocean_extraversion
    base.ocean_agreeableness = profile.ocean_agreeableness
    base.ocean_neuroticism = profile.ocean_neuroticism
    base.ocean_source = profile.ocean_source
    base.ocean_confidence = profile.ocean_confidence
    base.ocean_deviation_score = profile.ocean_deviation_score
    base.rewrite_count = profile.rewrite_count

    # Narrative status
    if any([
        base.persona,
        base.professional_persona,
        base.sports_persona,
        base.arts_persona,
        base.travel_persona,
        base.culinary_persona,
        base.cultural_background,
        base.skills_and_expertise,
        base.hobbies_and_interests,
        base.career_goals_and_ambitions,
    ]):
        base.narrative_status = "generated"
    else:
        base.narrative_status = "mock"

    return base
