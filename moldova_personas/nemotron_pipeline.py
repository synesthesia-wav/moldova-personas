"""
Nemotron-style compound pipeline for Moldova personas.

Implements the 3-stage approach:
  Stage A: Build persona core (structured, census-grounded)
  Stage B: Generate context fields (long narratives)
  Stage C: Generate 6 short personas from context

With validators and retry mechanisms.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable, Tuple
from enum import Enum
import uuid

from .generator import PersonaGenerator
from .models import Persona
from .llm_client import create_llm_client, LLMClient, GenerationConfig
from .soft_validators import SoftValidators, validate_persona_with_soft_checks


class ValidationError(Exception):
    """Raised when a validation check fails."""
    pass


class ValidationSeverity(Enum):
    HARD = "hard"  # Must fix - deterministic failure
    SOFT = "soft"  # Should fix - LLM judge


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    field: str
    message: str
    severity: ValidationSeverity
    score: Optional[float] = None  # 0-5 for soft validators


@dataclass
class PersonaCore:
    """Stage A: Structured core - single source of truth."""
    uuid: str
    # Demographics
    sex: str
    age: int
    marital_status: str
    education_level: str
    occupation: str
    # Geography
    region: str
    raion: str
    locality: str
    country: str = "Moldova"
    # Moldova-specific
    ethnicity: str = ""
    religion: str = ""
    employment_status: str = ""
    migration_background: str = ""


@dataclass
class ContextFields:
    """Stage B: Long narratives - shared latent story."""
    cultural_background: str = ""
    skills_and_expertise: str = ""
    skills_and_expertise_list: List[str] = field(default_factory=list)
    hobbies_and_interests: str = ""
    hobbies_and_interests_list: List[str] = field(default_factory=list)
    career_goals_and_ambitions: str = ""


@dataclass
class ShortPersonas:
    """Stage C: 6 short persona variants."""
    professional_persona: str = ""
    sports_persona: str = ""
    arts_persona: str = ""
    travel_persona: str = ""
    culinary_persona: str = ""
    persona: str = ""  # General "essence" statement


@dataclass
class NemotronPersona:
    """Complete Nemotron-style persona record (20 fields)."""
    # Identity
    uuid: str
    
    # 6 Short personas (Stage C)
    professional_persona: str
    sports_persona: str
    arts_persona: str
    travel_persona: str
    culinary_persona: str
    persona: str  # General essence
    
    # Context fields (Stage B)
    cultural_background: str
    skills_and_expertise: str
    skills_and_expertise_list: str  # Stringified list
    hobbies_and_interests: str
    hobbies_and_interests_list: str  # Stringified list
    career_goals_and_ambitions: str
    
    # Demographics (Stage A)
    sex: str
    age: int
    marital_status: str
    education_level: str
    occupation: str
    municipality: str  # maps to raion
    state: str  # maps to region
    country: str
    
    # Additional Moldova-specific
    locality: str = ""  # Village/city within raion
    ethnicity: str = ""
    religion: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_stages(
        cls,
        core: PersonaCore,
        context: ContextFields,
        personas: ShortPersonas
    ) -> "NemotronPersona":
        """Construct from pipeline stages."""
        return cls(
            uuid=core.uuid,
            professional_persona=personas.professional_persona,
            sports_persona=personas.sports_persona,
            arts_persona=personas.arts_persona,
            travel_persona=personas.travel_persona,
            culinary_persona=personas.culinary_persona,
            persona=personas.persona,
            cultural_background=context.cultural_background,
            skills_and_expertise=context.skills_and_expertise,
            skills_and_expertise_list=str(context.skills_and_expertise_list),
            hobbies_and_interests=context.hobbies_and_interests,
            hobbies_and_interests_list=str(context.hobbies_and_interests_list),
            career_goals_and_ambitions=context.career_goals_and_ambitions,
            sex=core.sex,
            age=core.age,
            marital_status=core.marital_status,
            education_level=core.education_level,
            occupation=core.occupation,
            municipality=core.raion,
            state=core.region,
            country=core.country,
            locality=core.locality,
            ethnicity=core.ethnicity,
            religion=core.religion
        )


class HardValidators:
    """Deterministic validators - fast, no LLM needed."""
    
    @staticmethod
    def validate_romanian(text: str, field: str) -> ValidationResult:
        """Check for Romanian language (basic heuristic)."""
        # Common Romanian diacritics and words
        romanian_markers = ['ă', 'â', 'î', 'ș', 'ț', 'și', 'care', 'pentru', 'din', 'cu']
        text_lower = text.lower()
        
        # Check for obvious non-Romanian markers
        non_romanian = ['the ', 'and ', 'with ', 'from ', 'for ', 'that ', 'this ']
        english_count = sum(1 for marker in non_romanian if marker in text_lower)
        
        if english_count >= 3:
            return ValidationResult(
                passed=False,
                field=field,
                message=f"Field appears to contain English text ({english_count} markers)",
                severity=ValidationSeverity.HARD
            )
        return ValidationResult(passed=True, field=field, message="OK", severity=ValidationSeverity.HARD)
    
    @staticmethod
    def validate_no_pii(text: str, field: str) -> ValidationResult:
        """Check for PII patterns."""
        # Email pattern
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            return ValidationResult(
                passed=False,
                field=field,
                message="Email address detected",
                severity=ValidationSeverity.HARD
            )
        
        # Phone number pattern (Moldovan)
        if re.search(r'\b0\d{8}\b|\+373\d{8}\b', text):
            return ValidationResult(
                passed=False,
                field=field,
                message="Phone number detected",
                severity=ValidationSeverity.HARD
            )
        
        # ID number pattern
        if re.search(r'\b\d{13}\b', text):
            return ValidationResult(
                passed=False,
                field=field,
                message="Potential ID number detected",
                severity=ValidationSeverity.HARD
            )
        
        return ValidationResult(passed=True, field=field, message="OK", severity=ValidationSeverity.HARD)
    
    @staticmethod
    def validate_length(text: str, field: str, min_len: int, max_len: int) -> ValidationResult:
        """Check text length bounds."""
        if len(text) < min_len:
            return ValidationResult(
                passed=False,
                field=field,
                message=f"Too short ({len(text)} chars, min {min_len})",
                severity=ValidationSeverity.HARD
            )
        if len(text) > max_len:
            return ValidationResult(
                passed=False,
                field=field,
                message=f"Too long ({len(text)} chars, max {max_len})",
                severity=ValidationSeverity.HARD
            )
        return ValidationResult(passed=True, field=field, message="OK", severity=ValidationSeverity.HARD)
    
    @staticmethod
    def validate_age_consistency(core: PersonaCore, context: ContextFields) -> ValidationResult:
        """Check age is consistent with career goals and background."""
        career_text = context.career_goals_and_ambitions.lower()
        
        # Retired but has career ambitions
        if core.age >= 65 and any(word in career_text for word in ['promovare', 'avansare', 'manager', 'director']):
            return ValidationResult(
                passed=False,
                field="career_goals_and_ambitions",
                message=f"Age {core.age} inconsistent with career ambitions",
                severity=ValidationSeverity.HARD
            )
        
        # Too young for certain career goals
        if core.age < 22 and any(word in career_text for word in ['pensionare', 'pensie']):
            return ValidationResult(
                passed=False,
                field="career_goals_and_ambitions",
                message=f"Age {core.age} too young for retirement plans",
                severity=ValidationSeverity.HARD
            )
        
        return ValidationResult(passed=True, field="career_goals", message="OK", severity=ValidationSeverity.HARD)


class NemotronPipeline:
    """Nemotron-style compound generation pipeline."""
    
    # Length bounds (chars)
    LENGTH_BOUNDS = {
        'persona': (50, 300),  # General essence - concise
        'professional_persona': (100, 400),
        'sports_persona': (100, 400),
        'arts_persona': (100, 400),
        'travel_persona': (100, 400),
        'culinary_persona': (100, 400),
        'cultural_background': (400, 1500),
        'skills_and_expertise': (200, 800),
        'hobbies_and_interests': (200, 800),
        'career_goals_and_ambitions': (150, 600),
    }
    
    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 3):
        self.llm = llm_client or create_llm_client("mock")
        self.max_retries = max_retries
        self.hard_validators = HardValidators()
    
    async def generate(
        self,
        persona: Persona,
        name: str,
        run_soft_validation: bool = False
    ) -> Tuple[NemotronPersona, Optional[dict]]:
        """
        Generate a complete Nemotron-style persona.
        
        Args:
            persona: Base Moldova persona
            name: Person's name
            run_soft_validation: If True, run LLM-as-judge validation
            
        Returns:
            Tuple of (NemotronPersona, validation_report or None)
        """
        # Stage A: Build core
        core = self._build_core(persona, name)
        
        # Stage B: Generate context fields with retry
        context = await self._generate_context_with_retry(core)
        
        # Stage C: Generate short personas with retry
        short_personas = await self._generate_personas_with_retry(core, context)
        
        # Assemble final record
        nemotron_persona = NemotronPersona.from_stages(core, context, short_personas)
        
        # Optional: Soft validation
        validation_report = None
        if run_soft_validation:
            validation_report = await validate_persona_with_soft_checks(
                nemotron_persona, self.llm
            )
        
        return nemotron_persona, validation_report
    
    def _build_core(self, persona: Persona, name: str) -> PersonaCore:
        """Stage A: Build structured core from Moldova persona."""
        # Extract name parts
        name_parts = name.split()
        first_name = name_parts[0] if name_parts else name
        
        return PersonaCore(
            uuid=str(uuid.uuid4()),
            sex=persona.sex,
            age=persona.age,
            marital_status=persona.marital_status,
            education_level=persona.education_level,
            occupation=persona.occupation,
            region=persona.region,
            raion=persona.district,
            locality=persona.city or persona.district,  # Fallback to district
            country="Moldova",
            ethnicity=getattr(persona, 'ethnicity', ''),
            religion=getattr(persona, 'religion', ''),
            employment_status=getattr(persona, 'employment_status', ''),
            migration_background=getattr(persona, 'migration_background', '')
        )
    
    async def _generate_context_with_retry(self, core: PersonaCore) -> ContextFields:
        """Stage B: Generate context fields with validation and retry."""
        for attempt in range(self.max_retries):
            try:
                context = await self._generate_context_fields(core)
                self._validate_context_hard(context)
                return context
            except ValidationError as e:
                if attempt == self.max_retries - 1:
                    raise
                # Retry with feedback
                continue
        raise ValidationError("Failed to generate valid context fields")
    
    async def _generate_context_fields(self, core: PersonaCore) -> ContextFields:
        """Generate long narrative context fields."""
        # Build prompt for context fields
        prompt = self._build_context_prompt(core)
        
        config = GenerationConfig(max_tokens=1500, temperature=0.7)
        response = await asyncio.to_thread(self.llm.generate, prompt, config)
        
        # Parse response
        return self._parse_context_response(response, core)
    
    def _build_context_prompt(self, core: PersonaCore) -> str:
        """Build prompt for Stage B - context fields."""
        return f"""Generează câmpurile de context pentru o persoană din Moldova.

DATE DEMOGRAFICE:
- Nume: {core.first_name if hasattr(core, 'first_name') else 'Persoană'}
- Vârstă: {core.age} ani
- Sex: {core.sex}
- Ocupație: {core.occupation}
- Educație: {core.education_level}
- Statut marital: {core.marital_status}
- Localitate: {core.locality}, {core.raion}, {core.region}
- Etnie: {core.ethnicity or 'Nespecificat'}
- Religie: {core.religion or 'Nespecificat'}

GENEREAZĂ următoarele câmpuri în limba ROMÂNĂ (cu diacritice):

1. CULTURAL_BACKGROUND (400-1500 caractere):
   - Context cultural și de viață din localitatea/raionul menționat
   - Valori, tradiții, mod de viață specific zonei
   - Influențe culturale relevante pentru Moldova

2. SKILLS_AND_EXPERTISE (200-800 caractere):
   - Abilități profesionale și personale
   - Expertiză relevantă pentru ocupația {core.occupation}
   - Competențe tehnice și soft skills

3. SKILLS_AND_EXPERTISE_LIST:
   - Lista abilităților în format: ['Abilitatea 1', 'Abilitatea 2', 'Abilitatea 3']
   - 5-8 abilități specifice

4. HOBBIES_AND_INTERESTS (200-800 caractere):
   - Pasiuni și interese în timpul liber
   - Activități de weekend
   - Interese culturale, sportive, sociale

5. HOBBIES_AND_INTERESTS_LIST:
   - Lista hobby-urilor în format: ['Hobby 1', 'Hobby 2', 'Hobby 3']
   - 4-6 hobby-uri realiste pentru vârsta și ocupația persoanei

6. CAREER_GOALS_AND_AMBITIONS (150-600 caractere):
   - Obiective profesionale realiste pentru vârsta {core.age}
   - Aspirații de carieră coerente cu {core.occupation}
   - Planuri de dezvoltare profesională

RĂSPUNS FORMAT JSON:
{{
  "cultural_background": "...",
  "skills_and_expertise": "...",
  "skills_and_expertise_list": ['...', '...'],
  "hobbies_and_interests": "...",
  "hobbies_and_interests_list": ['...', '...'],
  "career_goals_and_ambitions": "..."
}}

IMPORTANT:
- Toate câmpurile în ROMÂNĂ cu diacritice (ă, â, î, ș, ț)
- Fără adrese exacte sau date personale reale
- Coerență totală cu vârsta {core.age} și ocupația {core.occupation}
- Specificitate pentru regiunea {core.region}, raionul {core.raion}
"""
    
    def _parse_context_response(self, response: str, core: PersonaCore) -> ContextFields:
        """Parse LLM response into ContextFields."""
        # Extract JSON from response
        try:
            # Find JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Parse lists
            skills_list = data.get('skills_and_expertise_list', [])
            if isinstance(skills_list, str):
                skills_list = eval(skills_list) if skills_list.startswith('[') else [skills_list]
            
            hobbies_list = data.get('hobbies_and_interests_list', [])
            if isinstance(hobbies_list, str):
                hobbies_list = eval(hobbies_list) if hobbies_list.startswith('[') else [hobbies_list]
            
            return ContextFields(
                cultural_background=data.get('cultural_background', ''),
                skills_and_expertise=data.get('skills_and_expertise', ''),
                skills_and_expertise_list=skills_list,
                hobbies_and_interests=data.get('hobbies_and_interests', ''),
                hobbies_and_interests_list=hobbies_list,
                career_goals_and_ambitions=data.get('career_goals_and_ambitions', '')
            )
        except (json.JSONDecodeError, SyntaxError) as e:
            raise ValidationError(f"Failed to parse context response: {e}")
    
    def _validate_context_hard(self, context: ContextFields) -> None:
        """Run hard validators on context fields."""
        validations = [
            self.hard_validators.validate_romanian(context.cultural_background, "cultural_background"),
            self.hard_validators.validate_no_pii(context.cultural_background, "cultural_background"),
            self.hard_validators.validate_length(
                context.cultural_background, "cultural_background",
                *self.LENGTH_BOUNDS['cultural_background']
            ),
            self.hard_validators.validate_romanian(context.skills_and_expertise, "skills_and_expertise"),
            self.hard_validators.validate_length(
                context.skills_and_expertise, "skills_and_expertise",
                *self.LENGTH_BOUNDS['skills_and_expertise']
            ),
            self.hard_validators.validate_romanian(context.hobbies_and_interests, "hobbies_and_interests"),
            self.hard_validators.validate_length(
                context.hobbies_and_interests, "hobbies_and_interests",
                *self.LENGTH_BOUNDS['hobbies_and_interests']
            ),
            self.hard_validators.validate_romanian(context.career_goals_and_ambitions, "career_goals"),
            self.hard_validators.validate_length(
                context.career_goals_and_ambitions, "career_goals",
                *self.LENGTH_BOUNDS['career_goals_and_ambitions']
            ),
        ]
        
        failures = [v for v in validations if not v.passed]
        if failures:
            messages = "; ".join([f"{v.field}: {v.message}" for v in failures])
            raise ValidationError(f"Context validation failed: {messages}")
    
    async def _generate_personas_with_retry(
        self,
        core: PersonaCore,
        context: ContextFields
    ) -> ShortPersonas:
        """Stage C: Generate short personas with validation and retry."""
        for attempt in range(self.max_retries):
            try:
                personas = await self._generate_short_personas(core, context)
                self._validate_personas_hard(personas, core)
                return personas
            except ValidationError as e:
                if attempt == self.max_retries - 1:
                    raise
                continue
        raise ValidationError("Failed to generate valid short personas")
    
    async def _generate_short_personas(
        self,
        core: PersonaCore,
        context: ContextFields
    ) -> ShortPersonas:
        """Generate 6 short persona variants from context."""
        prompt = self._build_personas_prompt(core, context)
        
        config = GenerationConfig(max_tokens=1200, temperature=0.7)
        response = await asyncio.to_thread(self.llm.generate, prompt, config)
        
        return self._parse_personas_response(response, core)
    
    def _build_personas_prompt(self, core: PersonaCore, context: ContextFields) -> str:
        """Build prompt for Stage C - short personas."""
        return f"""Generează 6 variante scurte de persona pe baza contextului.

DATE DE BAZĂ:
- Nume: {core.first_name if hasattr(core, 'first_name') else 'Persoană'}
- Vârstă: {core.age} ani, Sex: {core.sex}
- Ocupație: {core.occupation}, Educație: {core.education_level}
- Localitate: {core.locality}, {core.raion}

CONTEXT (latent story):
Cultural Background: {context.cultural_background[:200]}...
Skills: {', '.join(context.skills_and_expertise_list[:3])}
Hobbies: {', '.join(context.hobbies_and_interests_list[:3])}
Career: {context.career_goals_and_ambitions[:150]}...

GENEREAZĂ 6 câmpuri PERSONA scurte (1-3 propoziții fiecare):

1. PERSONA (50-300 caractere):
   - O "esență" concisă a persoanei
   - Începe cu numele și vârsta
   - Personalitatea și valorile principale

2. PROFESSIONAL_PERSONA (100-400 caractere):
   - Profil profesional
   - Începe cu numele
   - Rolul ca {core.occupation}
   - Abordarea muncii și valorile profesionale

3. SPORTS_PERSONA (100-400 caractere):
   - Persoana în context sportiv/activ
   - Începe cu numele
   - Activități fizice preferate
   - Atitudinea față de sport

4. ARTS_PERSONA (100-400 caractere):
   - Persoana în context cultural/artistic
   - Începe cu numele
   - Interese artistice
   - Participarea la viața culturală

5. TRAVEL_PERSONA (100-400 caractere):
   - Persoana călător
   - Începe cu numele
   - Destinații preferate sau dorințe
   - Stilul de călătorie

6. CULINARY_PERSONA (100-400 caractere):
   - Persoana în bucătărie
   - Începe cu numele
   - Preferințe culinare
   - Bucate tradiționale sau internaționale

RĂSPUNS FORMAT JSON:
{{
  "persona": "Maria, 34 ani, este...",
  "professional_persona": "Maria lucrează ca...",
  "sports_persona": "În timpul liber, Maria...",
  "arts_persona": "Maria apreciază...",
  "travel_persona": "Călătoriile sunt...",
  "culinary_persona": "La masă, Maria preferă..."
}}

IMPORTANT:
- Fiecare câmp începe cu numele persoanei
- Coerență totală cu contextul de mai sus
- Limba ROMÂNĂ cu diacritice
- Fără contradicții între cele 6 variante
- Toate cele 6 să reflecte ACEEAși persoană
"""
    
    def _parse_personas_response(self, response: str, core: PersonaCore) -> ShortPersonas:
        """Parse LLM response into ShortPersonas."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return ShortPersonas(
                persona=data.get('persona', ''),
                professional_persona=data.get('professional_persona', ''),
                sports_persona=data.get('sports_persona', ''),
                arts_persona=data.get('arts_persona', ''),
                travel_persona=data.get('travel_persona', ''),
                culinary_persona=data.get('culinary_persona', '')
            )
        except (json.JSONDecodeError, SyntaxError) as e:
            raise ValidationError(f"Failed to parse personas response: {e}")
    
    def _validate_personas_hard(self, personas: ShortPersonas, core: PersonaCore) -> None:
        """Run hard validators on short personas."""
        fields = {
            'persona': personas.persona,
            'professional_persona': personas.professional_persona,
            'sports_persona': personas.sports_persona,
            'arts_persona': personas.arts_persona,
            'travel_persona': personas.travel_persona,
            'culinary_persona': personas.culinary_persona,
        }
        
        validations = []
        for field_name, text in fields.items():
            validations.append(self.hard_validators.validate_romanian(text, field_name))
            validations.append(self.hard_validators.validate_no_pii(text, field_name))
            validations.append(self.hard_validators.validate_length(
                text, field_name, *self.LENGTH_BOUNDS[field_name]
            ))
        
        failures = [v for v in validations if not v.passed]
        if failures:
            messages = "; ".join([f"{v.field}: {v.message}" for v in failures])
            raise ValidationError(f"Personas validation failed: {messages}")


async def generate_nemotron_personas(
    personas: List[Persona],
    names: List[str],
    provider: str = "mock",
    api_key: Optional[str] = None,
    model: str = "qwen-turbo",
    max_concurrent: int = 5,
    run_soft_validation: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate Nemotron-style personas from Moldova personas.
    
    Args:
        personas: List of Moldova Persona objects
        names: List of names (same length as personas)
        provider: LLM provider ("dashscope", "openai", "mock")
        api_key: API key for the provider
        model: Model name
        max_concurrent: Max concurrent LLM calls
        run_soft_validation: If True, run LLM-as-judge validation
        
    Returns:
        List of dicts with 'persona' and optional 'validation' keys
    """
    if provider == "mock":
        llm_client = create_llm_client("mock")
    else:
        llm_client = create_llm_client(provider, api_key=api_key, model=model)
    pipeline = NemotronPipeline(llm_client=llm_client)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_one(persona: Persona, name: str) -> Dict[str, Any]:
        async with semaphore:
            nemotron_persona, validation = await pipeline.generate(
                persona, name, run_soft_validation
            )
            result = {"persona": nemotron_persona}
            if validation:
                result["validation"] = validation
            return result
    
    tasks = [generate_one(p, n) for p, n in zip(personas, names)]
    return await asyncio.gather(*tasks)
