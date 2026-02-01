"""
Soft validators using LLM-as-judge for consistency scoring.

Implements Nemotron-style quality evaluation:
- Coherence between persona variants
- Demographic consistency
- Cultural authenticity
- Stereotype detection
"""

import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from enum import Enum

from .llm_client import LLMClient, GenerationConfig

if TYPE_CHECKING:
    from .nemotron_pipeline import NemotronPersona


class ValidationAspect(Enum):
    """Aspects of persona quality to validate."""
    COHERENCE = "coherence"  # Do all 6 personas agree?
    DEMOGRAPHIC = "demographic"  # Age/sex/education consistent?
    CULTURAL = "cultural"  # Matches region/locality?
    OCCUPATION = "occupation"  # Skills match job?
    STEREOTYPE = "stereotype"  # No caricatures?


@dataclass
class SoftValidationResult:
    """Result of soft validation."""
    aspect: ValidationAspect
    score: float  # 0-5 scale
    passed: bool  # score >= 3.0
    feedback: str  # Specific improvement suggestions
    field: Optional[str] = None  # Which field failed (if any)


class SoftValidators:
    """
    LLM-as-judge validators for persona quality.
    
    These are "soft" because they use LLM judgment rather than
    deterministic rules. They're slower but catch nuanced issues.
    """
    
    PASS_THRESHOLD = 3.0  # Score >= 3.0 is passing
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.config = GenerationConfig(max_tokens=500, temperature=0.3)
    
    async def validate_full(self, persona: Any) -> List[SoftValidationResult]:
        """
        Run all soft validators on a persona.
        
        Returns list of results - all must pass for acceptance.
        """
        tasks = [
            self.validate_coherence(persona),
            self.validate_demographic_consistency(persona),
            self.validate_cultural_authenticity(persona),
            self.validate_occupation_match(persona),
            self.validate_no_stereotypes(persona),
        ]
        return await asyncio.gather(*tasks)
    
    async def validate_coherence(self, persona: Any) -> SoftValidationResult:
        """
        Check that all 6 persona variants agree with each other.
        
        Nemotron key insight: coherence comes from shared context.
        We check if short personas contradict each other.
        """
        prompt = f"""Evaluează coerența dintre variantele de persona ale aceleiași persoane.

DATE:
- Vârstă: {persona.age} ani
- Ocupație: {persona.occupation}
- Educație: {persona.education_level}

PERSONA GENERALĂ:
{persona.persona}

PROFESIONALĂ:
{persona.professional_persona}

SPORT:
{persona.sports_persona}

ARTĂ:
{persona.arts_persona}

CĂLĂTORII:
{persona.travel_persona}

CULINARĂ:
{persona.culinary_persona}

ÎNTREBARE: Cele 6 variante de mai sus descriu ACEEAȘI persoană coerent?

Scalează 0-5:
5 = Perfect coerent, aceeași voce/personalitate în toate
4 = Mică variație de ton, dar aceeași persoană
3 = Unele mici contradicții (de exemplu, activ vs sedentar)
2 = Contradicții notabile (de exemplu, introvertit în unele, extrovertit în altele)
1 = Contradicții majore
0 = Par persoane complet diferite

RĂSPUNS FORMAT JSON:
{{
  "score": 4,
  "feedback": "Descriere specifică a problemei sau 'Coerent' dacă e OK"
}}

Răspunde doar cu JSON, fără alt text."""
        
        try:
            response = await asyncio.to_thread(self.llm.generate, prompt, self.config)
            result = json.loads(self._extract_json(response))
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback")
            
            return SoftValidationResult(
                aspect=ValidationAspect.COHERENCE,
                score=score,
                passed=score >= self.PASS_THRESHOLD,
                feedback=feedback
            )
        except Exception as e:
            return SoftValidationResult(
                aspect=ValidationAspect.COHERENCE,
                score=0,
                passed=False,
                feedback=f"Validation error: {e}"
            )
    
    async def validate_demographic_consistency(self, persona: Any) -> SoftValidationResult:
        """
        Check that narrative content matches demographics.
        
        Examples:
        - Age 18 shouldn't have 20 years work experience
        - Age 70 shouldn't have career promotion goals
        - Education should match occupation complexity
        """
        prompt = f"""Evaluează consistența demografică a descrierii.

DATE DEMOGRAFICE:
- Vârstă: {persona.age} ani
- Sex: {persona.sex}
- Educație: {persona.education_level}
- Ocupație: {persona.occupation}
- Statut marital: {persona.marital_status}

CONȚINUT DE VERIFICAT:
Cultural Background: {persona.cultural_background[:300]}...

Skills: {persona.skills_and_expertise[:300]}...

Career Goals: {persona.career_goals_and_ambitions}

ÎNTREBARE: Conținutul este realist pentru o persoană de {persona.age} ani, {persona.sex.lower()}, cu educație {persona.education_level.lower()}, lucrând ca {persona.occupation.lower()}?

Scalează 0-5:
5 = Perfect realist, vârsta și experiența se potrivesc perfect
4 = Realist, mici detalii care ar putea fi mai specifice
3 = Acceptabil, dar câteva elemente puțin probabil pentru vârstă
2 = Probleme notabile (ex: prea multă experiență pentru vârstă)
1 = Contradicții majore (ex: pensionar cu ambiții de promovare)
0 = Complet nerealist

RĂSPUNS FORMAT JSON:
{{
  "score": 4,
  "feedback": "Descriere specifică sau 'Consistent'"
}}

Răspunde doar cu JSON."""
        
        try:
            response = await asyncio.to_thread(self.llm.generate, prompt, self.config)
            result = json.loads(self._extract_json(response))
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback")
            
            return SoftValidationResult(
                aspect=ValidationAspect.DEMOGRAPHIC,
                score=score,
                passed=score >= self.PASS_THRESHOLD,
                feedback=feedback
            )
        except Exception as e:
            return SoftValidationResult(
                aspect=ValidationAspect.DEMOGRAPHIC,
                score=0,
                passed=False,
                feedback=f"Validation error: {e}"
            )
    
    async def validate_cultural_authenticity(self, persona: Any) -> SoftValidationResult:
        """
        Check that cultural references match Moldova and specific region.
        
        Examples:
        - References to Moldovan traditions, not generic
        - Regional specifics (Nord vs Sud vs Gagauzia)
        - Romanian language usage
        """
        prompt = f"""Evaluează autenticitatea culturală pentru Moldova.

CONTEXT GEOGRAFIC:
- Țară: {persona.country}
- Regiune: {persona.state}
- Raion: {persona.municipality}
- Localitate: {persona.locality}
- Etnie: {persona.ethnicity}
- Religie: {persona.religion}

CONȚINUT DE VERIFICAT:
Cultural Background: {persona.cultural_background}

Hobbies: {persona.hobbies_and_interests}

Culinary: {persona.culinary_persona}

ÎNTREBARE: Conținutul reflectă autentic cultura și contextul din {persona.locality}, {persona.municipality}, Moldova?

Scalează 0-5:
5 = Autentic moldovenesc, referințe specifice locale (tradiții, mâncăruri, obiceiuri)
4 = Moldovenesc general, câteva elemente specifice regiunii
3 = Generic est-european, puține elemente moldovenești specifice
2 = Prea generic, ar putea fi din orice țară
1 = Elemente care nu se potrivesc cu Moldova
0 = Cultural greșit (ex: referințe la țări/zone complet diferite)

RĂSPUNS FORMAT JSON:
{{
  "score": 4,
  "feedback": "Elementele autentice identificate sau lipsurile"
}}

Răspunde doar cu JSON."""
        
        try:
            response = await asyncio.to_thread(self.llm.generate, prompt, self.config)
            result = json.loads(self._extract_json(response))
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback")
            
            return SoftValidationResult(
                aspect=ValidationAspect.CULTURAL,
                score=score,
                passed=score >= self.PASS_THRESHOLD,
                feedback=feedback
            )
        except Exception as e:
            return SoftValidationResult(
                aspect=ValidationAspect.CULTURAL,
                score=0,
                passed=False,
                feedback=f"Validation error: {e}"
            )
    
    async def validate_occupation_match(self, persona: Any) -> SoftValidationResult:
        """
        Check that skills and background match occupation.
        
        Examples:
        - Doctor should have medical skills
        - Farmer should have agricultural knowledge
        """
        prompt = f"""Evaluează potrivirea competențelor cu ocupația.

OCUPAȚIE: {persona.occupation}
EDUCAȚIE: {persona.education_level}
VÂRSTĂ: {persona.age} ani

SKILLS DECLARATE:
{persona.skills_and_expertise}

LISTĂ SKILLS: {persona.skills_and_expertise_list}

PROFIL PROFESIONAL:
{persona.professional_persona}

ÎNTREBARE: Competențele descrise sunt adecvate și plauzibile pentru un {persona.occupation.lower()} cu educație {persona.education_level.lower()}?

Scalează 0-5:
5 = Skills perfect potrivite, expertiză credibilă pentru rol
4 = Skills bune, mici omisiuni sau adăugiri inutile
3 = Acceptabil, dar câteva skills nu se potrivesc bine
2 = Skills parțial potrivite, unele neconectate cu ocupația
1 = Skills în mare parte nepotrivite
0 = Complet nepotrivit (ex: doctor cu skills de programare)

RĂSPUNS FORMAT JSON:
{{
  "score": 4,
  "feedback": "Analiza potrivirii skills cu ocupația"
}}

Răspunde doar cu JSON."""
        
        try:
            response = await asyncio.to_thread(self.llm.generate, prompt, self.config)
            result = json.loads(self._extract_json(response))
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback")
            
            return SoftValidationResult(
                aspect=ValidationAspect.OCCUPATION,
                score=score,
                passed=score >= self.PASS_THRESHOLD,
                feedback=feedback
            )
        except Exception as e:
            return SoftValidationResult(
                aspect=ValidationAspect.OCCUPATION,
                score=0,
                passed=False,
                feedback=f"Validation error: {e}"
            )
    
    async def validate_no_stereotypes(self, persona: Any) -> SoftValidationResult:
        """
        Check for harmful stereotypes or caricatures.
        
        Examples:
        - Gender stereotypes (women only cooking, men only driving)
        - Ethnic stereotypes
        - Rural vs urban bias
        """
        prompt = f"""Evaluează prezența stereotipurilor sau caricaturizării.

DATE:
- Sex: {persona.sex}
- Vârstă: {persona.age}
- Etnie: {persona.ethnicity}
- Regiune: {persona.state} ({persona.municipality})
- Ocupație: {persona.occupation}

CONȚINUT DE VERIFICAT:
Persona: {persona.persona}

Profesional: {persona.professional_persona}

Cultural: {persona.cultural_background[:400]}...

Hobbies: {persona.hobbies_and_interests[:400]}...

ÎNTREBARE: Conținutul conține stereotipuri dăunătoare sau caricaturizări? Este respectuos și nuanțat?

Scalează 0-5 (5 = cel mai bun):
5 = Descriere nuanțată și respectuoasă, fără stereotipuri
4 = În general OK, poate câteva mici generalizări
3 = Unele stereotipuri ușoare (ex: "toate femeile gătesc")
2 = Stereotipuri notabile (ex: descrieri care reduc persoana la un singur trait)
1 = Caricaturizare clară
0 = Stereotipuri dăunătoare majore

RĂSPUNS FORMAT JSON:
{{
  "score": 4,
  "feedback": "Identificare stereotipuri sau confirmare respectuos"
}}

Răspunde doar cu JSON."""
        
        try:
            response = await asyncio.to_thread(self.llm.generate, prompt, self.config)
            result = json.loads(self._extract_json(response))
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback")
            
            return SoftValidationResult(
                aspect=ValidationAspect.STEREOTYPE,
                score=score,
                passed=score >= self.PASS_THRESHOLD,
                feedback=feedback
            )
        except Exception as e:
            return SoftValidationResult(
                aspect=ValidationAspect.STEREOTYPE,
                score=0,
                passed=False,
                feedback=f"Validation error: {e}"
            )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        import re
        # Try to find JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group()
        return text
    
    def generate_report(self, results: List[SoftValidationResult]) -> dict:
        """Generate a summary report of validation results."""
        total_score = sum(r.score for r in results)
        avg_score = total_score / len(results) if results else 0
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        return {
            "average_score": round(avg_score, 2),
            "total_score": round(total_score, 2),
            "passed": passed,
            "failed": failed,
            "accept": all(r.passed for r in results),
            "details": [
                {
                    "aspect": r.aspect.value,
                    "score": r.score,
                    "passed": r.passed,
                    "feedback": r.feedback
                }
                for r in results
            ]
        }


async def validate_persona_with_soft_checks(
    persona: Any,
    llm_client: LLMClient
) -> dict:
    """
    Convenience function to run all soft validations on a persona.
    
    Returns a complete report with scores and recommendations.
    """
    validators = SoftValidators(llm_client)
    results = await validators.validate_full(persona)
    return validators.generate_report(results)
