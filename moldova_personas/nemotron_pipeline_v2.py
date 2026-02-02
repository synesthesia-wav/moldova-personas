"""
Nemotron-style pipeline v2 with OCEAN personality framework.

Implements the two-layer artifact:
1. Structured persona object (with OCEAN + behavioral contract)
2. Narrative rendering consistent with structure

With score-and-rewrite loop for OCEAN consistency.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .models import Persona
from .llm_client import create_llm_client, LLMClient, GenerationConfig
from .ocean_framework import (
    OCEANProfile, 
    OCEANSampler,
    OCEANBehaviorMapper,
    OCEANTextAnalyzer,
    generate_ocean_guided_prompt
)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


@dataclass  
class NemotronPersonaV2:
    """
    Complete Nemotron-style persona with OCEAN personality.
    
    Two-layer structure:
    - Layer 1: Structured data (demographics + OCEAN + behavioral contract)
    - Layer 2: Narrative rendering (consistent with Layer 1)
    """
    # === LAYER 1: STRUCTURED ===
    uuid: str
    
    # Demographics
    sex: str
    age: int
    marital_status: str
    education_level: str
    occupation: str
    municipality: str  # raion
    state: str  # region
    locality: str
    country: str = "Moldova"
    ethnicity: str = ""
    religion: str = ""
    
    # OCEAN Personality (Layer 1 Core)
    ocean_openness: int = 50
    ocean_conscientiousness: int = 50
    ocean_extraversion: int = 50
    ocean_agreeableness: int = 50
    ocean_neuroticism: int = 50
    ocean_source: str = "sampled"
    ocean_confidence: float = 1.0
    
    # Behavioral Contract (Derived from OCEAN)
    behavior_risk_tolerance: str = "moderate"
    behavior_decision_style: str = "pragmatic"
    behavior_conflict_style: str = "compromising"
    behavior_social_pattern: str = "selective"
    behavior_novelty_seeking: str = "balanced"
    behavior_dissent_probability: float = 0.3
    behavior_complaint_likelihood: str = "moderate"
    behavior_planning_horizon: str = "medium_term"
    
    # === LAYER 2: NARRATIVE ===
    # 6 Short personas
    persona: str = ""  # General essence
    professional_persona: str = ""
    sports_persona: str = ""
    arts_persona: str = ""
    travel_persona: str = ""
    culinary_persona: str = ""
    
    # Context fields
    cultural_background: str = ""
    skills_and_expertise: str = ""
    skills_and_expertise_list: str = "[]"
    hobbies_and_interests: str = ""
    hobbies_and_interests_list: str = "[]"
    career_goals_and_ambitions: str = ""
    
    # Validation metadata
    ocean_deviation_score: Optional[float] = None  # How far narrative is from target OCEAN
    rewrite_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_ocean_profile(self) -> OCEANProfile:
        """Extract OCEAN profile."""
        return OCEANProfile(
            openness=self.ocean_openness,
            conscientiousness=self.ocean_conscientiousness,
            extraversion=self.ocean_extraversion,
            agreeableness=self.ocean_agreeableness,
            neuroticism=self.ocean_neuroticism,
            source=self.ocean_source,
            confidence=self.ocean_confidence
        )
    
    def get_behavioral_contract(self) -> Dict[str, Any]:
        """Extract behavioral contract."""
        return {
            "risk_tolerance": self.behavior_risk_tolerance,
            "decision_style": self.behavior_decision_style,
            "conflict_style": self.behavior_conflict_style,
            "social_pattern": self.behavior_social_pattern,
            "novelty_seeking": self.behavior_novelty_seeking,
            "will_say_no_probability": self.behavior_dissent_probability,
            "complaint_likelihood": self.behavior_complaint_likelihood,
            "planning_horizon": self.behavior_planning_horizon,
        }


class NemotronPipelineV2:
    """
    Nemotron pipeline with OCEAN personality and score-and-rewrite loop.
    
    Two-pass generation:
    Pass 1: Structure (demographics + OCEAN + behavioral contract)
    Pass 2: Narrative (consistent with structure)
    
    With OCEAN validation and rewrite loop.
    """
    
    def __init__(
        self, 
        llm_client: Optional[LLMClient] = None,
        ocean_tolerance: int = 15,  # Max acceptable deviation per trait
        max_rewrites: int = 3
    ):
        self.llm = llm_client or create_llm_client("mock")
        self.ocean_sampler = OCEANSampler()
        self.ocean_analyzer = OCEANTextAnalyzer()
        self.ocean_tolerance = ocean_tolerance
        self.max_rewrites = max_rewrites
        
        self.generation_config = GenerationConfig(
            max_tokens=1500,
            temperature=0.7
        )
        self.rewrite_config = GenerationConfig(
            max_tokens=1500,
            temperature=0.8  # Slightly higher for variation
        )
    
    async def generate(
        self,
        base_persona: Persona,
        name: str
    ) -> NemotronPersonaV2:
        """
        Generate complete Nemotron persona with OCEAN.
        
        Args:
            base_persona: Base Moldova persona with demographics
            name: Person's full name
            
        Returns:
            NemotronPersonaV2 with validated OCEAN consistency
        """
        # === PASS 1: STRUCTURE ===
        structured = self._generate_structure(base_persona, name)
        
        # === PASS 2: NARRATIVE ===
        # Initial generation
        narrative = await self._generate_narrative(structured, name)
        
        # === SCORE-AND-REWRITE LOOP ===
        # Validate OCEAN consistency and rewrite if needed
        for rewrite_attempt in range(self.max_rewrites):
            is_valid, deviation_info = await self._validate_ocean_consistency(
                structured, narrative
            )
            
            if is_valid:
                structured.ocean_deviation_score = deviation_info["max_deviation"]
                break
            
            if rewrite_attempt < self.max_rewrites - 1:
                # Rewrite with targeted corrections
                narrative = await self._rewrite_narrative(
                    structured, narrative, deviation_info
                )
                structured.rewrite_count += 1
            else:
                # Max rewrites reached, store deviation
                structured.ocean_deviation_score = deviation_info["max_deviation"]
        
        # Merge narrative into structured
        return self._merge_narrative(structured, narrative)
    
    def _generate_structure(
        self,
        base: Persona,
        name: str
    ) -> NemotronPersonaV2:
        """
        Pass 1: Generate structured layer.
        
        Includes:
        - Demographics
        - OCEAN profile (sampled conditioned on demographics)
        - Behavioral contract (derived from OCEAN)
        """
        # Sample OCEAN conditioned on demographics
        ocean = self.ocean_sampler.sample(
            age=base.age,
            sex=base.sex,
            education_level=base.education_level,
            occupation=base.occupation
        )
        
        # Generate behavioral contract
        contract = OCEANBehaviorMapper.generate_behavioral_contract(ocean)
        
        return NemotronPersonaV2(
            uuid=base.uuid,
            sex=base.sex,
            age=base.age,
            marital_status=base.marital_status,
            education_level=base.education_level,
            occupation=base.occupation,
            municipality=base.district,
            state=base.region,
            locality=base.city,
            country="Moldova",
            ethnicity=getattr(base, 'ethnicity', 'Moldovean'),
            religion=getattr(base, 'religion', 'Ortodox'),
            # OCEAN
            ocean_openness=ocean.openness,
            ocean_conscientiousness=ocean.conscientiousness,
            ocean_extraversion=ocean.extraversion,
            ocean_agreeableness=ocean.agreeableness,
            ocean_neuroticism=ocean.neuroticism,
            ocean_source=ocean.source,
            ocean_confidence=ocean.confidence,
            # Behavioral contract
            behavior_risk_tolerance=contract["risk_tolerance"],
            behavior_decision_style=contract["decision_style"],
            behavior_conflict_style=contract["conflict_style"],
            behavior_social_pattern=contract["social_pattern"],
            behavior_novelty_seeking=contract["novelty_seeking"],
            behavior_dissent_probability=contract["will_say_no_probability"],
            behavior_complaint_likelihood=contract["complaint_likelihood"],
            behavior_planning_horizon=contract["planning_horizon"],
        )
    
    async def _generate_narrative(
        self,
        structured: NemotronPersonaV2,
        name: str
    ) -> Dict[str, str]:
        """
        Pass 2: Generate narrative layer.
        
        Uses OCEAN-guided prompts for consistency.
        """
        ocean = structured.get_ocean_profile()
        contract = structured.get_behavioral_contract()
        
        # Generate ocean-guided prompt
        ocean_prompt = generate_ocean_guided_prompt(ocean, contract)
        
        # Generate context fields
        context = await self._generate_context_fields(structured, name, ocean_prompt)
        
        # Generate 6 short personas
        personas = await self._generate_short_personas(
            structured, name, context, ocean_prompt
        )
        
        return {
            **context,
            **personas
        }
    
    async def _generate_context_fields(
        self,
        structured: NemotronPersonaV2,
        name: str,
        ocean_prompt: str
    ) -> Dict[str, str]:
        """Generate long-form context fields with OCEAN guidance."""
        first_name = name.split()[0]
        
        prompt = f"""Generează câmpurile de context pentru o persoană din Moldova.

NUME: {name}
VÂRSTĂ: {structured.age} ani
SEX: {structured.sex}
OCUPAȚIE: {structured.occupation}
EDUCAȚIE: {structured.education_level}
LOCALITATE: {structured.locality}, {structured.municipality}

{ocean_prompt}

GENEREAZĂ în limba ROMÂNĂ (cu diacritice):

1. CULTURAL_BACKGROUND (400-1500 caractere):
   Context de viață din {structured.locality}, {structured.municipality}
   Valori și tradiții specifice zonei
   CUM această personalitate specifică se manifestă în acest context

2. SKILLS_AND_EXPERTISE (200-800 caractere):
   Abilități profesionale coerente cu {structured.occupation}
   Soft skills care reflectă personalitatea OCEAN
   Expertiză dobândită

3. SKILLS_LIST: ['Skill1', 'Skill2', ...] - 5-8 abilități

4. HOBBIES_AND_INTERESTS (200-800 caractere):
   Pasiuni care reflectă personalitatea OCEAN
   Activități de weekend coerente cu patternul social
   Interese autentice pentru această persoană

5. HOBBIES_LIST: ['Hobby1', 'Hobby2', ...] - 4-6 hobby-uri

6. CAREER_GOALS (150-600 caractere):
   Obiective realiste pentru vârsta {structured.age}
   Coerente cu {structured.occupation}
   Care reflectă toleranța la risc și orizontul de planificare

IMPORTANT:
- Toate câmpurile trebuie să REFLECTE personalitatea OCEAN descrisă mai sus
- Evită clișee pozitive - dacă profilul indică anxietate sau competitivitate, arată asta
- Fiecare aspect al personalității trebuie vizibil în activități și decizii

RĂSPUNS JSON:
{{
  "cultural_background": "...",
  "skills_and_expertise": "...",
  "skills_list": ["..."],
  "hobbies_and_interests": "...",
  "hobbies_list": ["..."],
  "career_goals": "..."
}}"""
        
        response = await asyncio.to_thread(
            self.llm.generate, prompt, self.generation_config
        )
        
        return self._parse_context_response(response)
    
    async def _generate_short_personas(
        self,
        structured: NemotronPersonaV2,
        name: str,
        context: Dict[str, str],
        ocean_prompt: str
    ) -> Dict[str, str]:
        """Generate 6 short persona variants with OCEAN guidance."""
        first_name = name.split()[0]
        
        background = context.get('cultural_background', '')[:300]
        hobbies = context.get('hobbies_and_interests', '')[:200]
        
        prompt = f"""Generează 6 variante scurte de persona.

NUME: {name} ({structured.age} ani, {structured.occupation})

{ocean_prompt}

CONTEXT GENERAT:
{background}

Hobbies: {hobbies}

GENEREAZĂ 6 câmpuri (1-3 propoziții fiecare):

1. PERSONA - esența generală, vocea unică a acestei persoane
2. PROFESSIONAL_PERSONA - la locul de muncă
3. SPORTS_PERSONA - în context sportiv/activ
4. ARTS_PERSONA - în context cultural
5. TRAVEL_PERSONA - ca călător
6. CULINARY_PERSONA - în bucătărie

REGULI:
- Fiecare variantă începe cu "{first_name}"
- Toate reflect ACEEAȘI personalitate OCEAN
- Nu evita trăsăturile negative - dacă e anxios, competitiv, sau rezervat, arată asta
- Coerență totală între cele 6 variante

RĂSPUNS JSON:
{{
  "persona": "{first_name}...",
  "professional_persona": "{first_name}...",
  "sports_persona": "{first_name}...",
  "arts_persona": "{first_name}...",
  "travel_persona": "{first_name}...",
  "culinary_persona": "{first_name}..."
}}"""
        
        response = await asyncio.to_thread(
            self.llm.generate, prompt, self.generation_config
        )
        
        return self._parse_personas_response(response)
    
    async def _validate_ocean_consistency(
        self,
        structured: NemotronPersonaV2,
        narrative: Dict[str, str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that narrative matches target OCEAN profile.
        
        Returns:
            (is_valid, deviation_info)
        """
        target = structured.get_ocean_profile()
        
        # Combine all narrative text
        combined_text = " ".join([
            narrative.get("persona", ""),
            narrative.get("professional_persona", ""),
            narrative.get("cultural_background", ""),
            narrative.get("hobbies_and_interests", "")
        ])
        
        # Infer OCEAN from text
        is_within, deviations, inferred = self.ocean_analyzer.calculate_deviation(
            target, combined_text, tolerance=self.ocean_tolerance
        )
        
        max_deviation = max(abs(d) for d in deviations.values())
        confidence = inferred.confidence

        rewrite_policy = "standard"
        if confidence < 0.35:
            # Low-signal text: avoid random rewrites
            is_within = True
            rewrite_policy = "skip_low_confidence"
        elif confidence < 0.6:
            # Medium confidence: only rewrite on large deviations
            if max_deviation <= 25:
                is_within = True
                rewrite_policy = "skip_medium_confidence"
            else:
                rewrite_policy = "allow_large_deviation"
        
        deviation_info = {
            "is_within_tolerance": is_within,
            "target": target.to_dict(),
            "inferred": inferred.to_dict(),
            "deviations": deviations,
            "max_deviation": max_deviation,
            "confidence": confidence,
            "rewrite_policy": rewrite_policy,
            "traits_to_adjust": [
                trait for trait, dev in deviations.items()
                if abs(dev) > self.ocean_tolerance
            ]
        }
        
        return is_within, deviation_info
    
    async def _rewrite_narrative(
        self,
        structured: NemotronPersonaV2,
        narrative: Dict[str, str],
        deviation_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Rewrite narrative to better match target OCEAN.
        
        Targeted rewrite based on deviation analysis.
        """
        traits_to_adjust = deviation_info["traits_to_adjust"]
        target = deviation_info["target"]
        inferred = deviation_info["inferred"]
        
        # Build targeted rewrite instructions
        adjustments = []
        for trait in traits_to_adjust:
            target_val = target.get(trait)
            inferred_val = inferred.get(trait)
            diff = target_val - inferred_val
            
            if trait == "openness":
                if diff > 0:
                    adjustments.append(f"CREȘTE Openness: adaugă mai multă curiozitate, creativitate, preferință pentru noutate (target: {target_val}, actual: {inferred_val})")
                else:
                    adjustments.append(f"SCADE Openness: fă persoana mai practică, tradițională, preferă rutina (target: {target_val}, actual: {inferred_val})")
            
            elif trait == "conscientiousness":
                if diff > 0:
                    adjustments.append(f"CREȘTE Conscientiousness: mai organizat, disciplinat, planificat (target: {target_val}, actual: {inferred_val})")
                else:
                    adjustments.append(f"SCADE Conscientiousness: mai spontan, flexibil, laid-back (target: {target_val}, actual: {inferred_val})")
            
            elif trait == "extraversion":
                if diff > 0:
                    adjustments.append(f"CREȘTE Extraversion: mai sociabil, energic, asertiv (target: {target_val}, actual: {inferred_val})")
                else:
                    adjustments.append(f"SCADE Extraversion: mai rezervat, introspectiv, quiet (target: {target_val}, actual: {inferred_val})")
            
            elif trait == "agreeableness":
                if diff > 0:
                    adjustments.append(f"CREȘTE Agreeableness: mai amabil, cooperant, empatic (target: {target_val}, actual: {inferred_val})")
                else:
                    adjustments.append(f"SCADE Agreeableness: mai direct, competitiv, critic (target: {target_val}, actual: {inferred_val})")
            
            elif trait == "neuroticism":
                if diff > 0:
                    adjustments.append(f"CREȘTE Neuroticism: mai anxios, stresat, emotional, precaut (target: {target_val}, actual: {inferred_val})")
                else:
                    adjustments.append(f"SCADE Neuroticism: mai calm, stabil, încrezător (target: {target_val}, actual: {inferred_val})")
        
        # Rewrite prompt
        prompt = f"""RESCRIE descrierile pentru a corecta devierea de personalitate.

AJUSTĂRI NECESARE:
{chr(10).join(f"- {adj}" for adj in adjustments)}

DESCRIERI ACTUALE:
PERSONA: {narrative.get('persona', '')}

PROFESIONAL: {narrative.get('professional_persona', '')}

CULTURAL: {narrative.get('cultural_background', '')[:400]}...

HOBBIES: {narrative.get('hobbies_and_interests', '')[:400]}...

INSTRUCȚIUNI:
1. Păstrează structura și informațiile de bază
2. Aplică AJUSTĂRILE de mai sus
3. Asigură coerență între toate variantele
4. Răspunde în format JSON cu aceleași câmpuri

RĂSPUNS JSON cu aceleași câmpuri: persona, professional_persona, sports_persona, arts_persona, travel_persona, culinary_persona, cultural_background, skills_and_expertise, skills_list, hobbies_and_interests, hobbies_list, career_goals"""
        
        response = await asyncio.to_thread(
            self.llm.generate, prompt, self.rewrite_config
        )
        
        # Try to parse and merge
        try:
            new_narrative = self._parse_full_response(response)
            # Keep any fields that weren't in rewrite
            for key in narrative:
                if key not in new_narrative or not new_narrative[key]:
                    new_narrative[key] = narrative[key]
            return new_narrative
        except:
            # If rewrite fails, return original
            return narrative
    
    def _merge_narrative(
        self,
        structured: NemotronPersonaV2,
        narrative: Dict[str, str]
    ) -> NemotronPersonaV2:
        """Merge narrative into structured persona."""
        structured.persona = narrative.get("persona", "")
        structured.professional_persona = narrative.get("professional_persona", "")
        structured.sports_persona = narrative.get("sports_persona", "")
        structured.arts_persona = narrative.get("arts_persona", "")
        structured.travel_persona = narrative.get("travel_persona", "")
        structured.culinary_persona = narrative.get("culinary_persona", "")
        structured.cultural_background = narrative.get("cultural_background", "")
        structured.skills_and_expertise = narrative.get("skills_and_expertise", "")
        structured.skills_and_expertise_list = str(narrative.get("skills_list", []))
        structured.hobbies_and_interests = narrative.get("hobbies_and_interests", "")
        structured.hobbies_and_interests_list = str(narrative.get("hobbies_list", []))
        structured.career_goals_and_ambitions = narrative.get("career_goals", "")
        
        return structured
    
    def _parse_context_response(self, response: str) -> Dict[str, str]:
        """Parse context fields from LLM response."""
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
            
            return {
                "cultural_background": data.get("cultural_background", ""),
                "skills_and_expertise": data.get("skills_and_expertise", ""),
                "skills_list": data.get("skills_list", []),
                "hobbies_and_interests": data.get("hobbies_and_interests", ""),
                "hobbies_list": data.get("hobbies_list", []),
                "career_goals": data.get("career_goals", "")
            }
        except:
            return {
                "cultural_background": "",
                "skills_and_expertise": "",
                "skills_list": [],
                "hobbies_and_interests": "",
                "hobbies_list": [],
                "career_goals": ""
            }
    
    def _parse_personas_response(self, response: str) -> Dict[str, str]:
        """Parse short personas from LLM response."""
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
            
            return {
                "persona": data.get("persona", ""),
                "professional_persona": data.get("professional_persona", ""),
                "sports_persona": data.get("sports_persona", ""),
                "arts_persona": data.get("arts_persona", ""),
                "travel_persona": data.get("travel_persona", ""),
                "culinary_persona": data.get("culinary_persona", "")
            }
        except:
            return {
                "persona": "",
                "professional_persona": "",
                "sports_persona": "",
                "arts_persona": "",
                "travel_persona": "",
                "culinary_persona": ""
            }
    
    def _parse_full_response(self, response: str) -> Dict[str, str]:
        """Parse full narrative response."""
        result = {}
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
            
            # Map all possible fields
            field_mapping = {
                "persona": "persona",
                "professional_persona": "professional_persona",
                "sports_persona": "sports_persona",
                "arts_persona": "arts_persona",
                "travel_persona": "travel_persona",
                "culinary_persona": "culinary_persona",
                "cultural_background": "cultural_background",
                "skills_and_expertise": "skills_and_expertise",
                "skills_list": "skills_list",
                "hobbies_and_interests": "hobbies_and_interests",
                "hobbies_list": "hobbies_list",
                "career_goals": "career_goals"
            }
            
            for key, mapped in field_mapping.items():
                if key in data:
                    result[mapped] = data[key]
            
        except:
            pass
        
        return result


async def generate_nemotron_v2_personas(
    personas: List[Persona],
    names: List[str],
    provider: str = "mock",
    api_key: Optional[str] = None,
    model: str = "qwen-turbo",
    max_concurrent: int = 5,
    ocean_tolerance: int = 15,
    max_rewrites: int = 3
) -> List[NemotronPersonaV2]:
    """
    Generate Nemotron v2 personas with OCEAN personality.
    
    Args:
        personas: List of base Moldova personas
        names: List of names
        provider: LLM provider
        api_key: API key
        model: Model name
        max_concurrent: Max concurrent LLM calls
        ocean_tolerance: Max acceptable OCEAN deviation per trait
        max_rewrites: Max rewrite attempts for OCEAN consistency
        
    Returns:
        List of NemotronPersonaV2 with validated OCEAN
    """
    if provider == "mock":
        llm_client = create_llm_client("mock")
    else:
        llm_client = create_llm_client(provider, api_key=api_key, model=model)
    
    pipeline = NemotronPipelineV2(
        llm_client=llm_client,
        ocean_tolerance=ocean_tolerance,
        max_rewrites=max_rewrites
    )
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_one(persona: Persona, name: str) -> NemotronPersonaV2:
        async with semaphore:
            return await pipeline.generate(persona, name)
    
    tasks = [generate_one(p, n) for p, n in zip(personas, names)]
    return await asyncio.gather(*tasks)
