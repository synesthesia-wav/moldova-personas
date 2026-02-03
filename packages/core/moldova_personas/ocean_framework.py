"""
OCEAN Big Five Personality Framework for personas.

Implements:
- OCEAN trait sampling (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- Demographic-conditioned sampling
- OCEAN-to-behavior mapping rules
- Big Five text inference for validation
- Score-and-rewrite loop for consistency

Based on research linking Big Five traits to behavioral patterns:
- Risk tolerance ↔ low N, high O
- Planning/discipline ↔ high C
- Social energy ↔ high E
- Conflict style ↔ A + N
"""

import random
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class Trait(Enum):
    """Big Five traits."""
    OPENNESS = "openness"  # O
    CONSCIENTIOUSNESS = "conscientiousness"  # C
    EXTRAVERSION = "extraversion"  # E
    AGREEABLENESS = "agreeableness"  # A
    NEUROTICISM = "neuroticism"  # N


@dataclass
class OCEANProfile:
    """
    Big Five personality profile (0-100 scale).
    
    Each trait represents:
    - Openness: Curiosity, creativity, preference for novelty
    - Conscientiousness: Organization, diligence, self-discipline
    - Extraversion: Sociability, assertiveness, positive emotionality
    - Agreeableness: Cooperation, trust, prosocial orientation
    - Neuroticism: Emotional instability, anxiety, negative emotionality
    """
    openness: int = 50
    conscientiousness: int = 50
    extraversion: int = 50
    agreeableness: int = 50
    neuroticism: int = 50
    
    # Metadata
    source: str = "sampled"  # "sampled", "inferred", "calibrated"
    confidence: float = 1.0  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "source": self.source,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCEANProfile":
        """Create from dictionary."""
        return cls(
            openness=data.get("openness", 50),
            conscientiousness=data.get("conscientiousness", 50),
            extraversion=data.get("extraversion", 50),
            agreeableness=data.get("agreeableness", 50),
            neuroticism=data.get("neuroticism", 50),
            source=data.get("source", "sampled"),
            confidence=data.get("confidence", 1.0)
        )
    
    def distance(self, other: "OCEANProfile") -> float:
        """Calculate Euclidean distance between two profiles."""
        return (
            (self.openness - other.openness) ** 2 +
            (self.conscientiousness - other.conscientiousness) ** 2 +
            (self.extraversion - other.extraversion) ** 2 +
            (self.agreeableness - other.agreeableness) ** 2 +
            (self.neuroticism - other.neuroticism) ** 2
        ) ** 0.5
    
    def is_within_tolerance(self, other: "OCEANProfile", tolerance: int = 10) -> bool:
        """Check if all traits are within tolerance."""
        return (
            abs(self.openness - other.openness) <= tolerance and
            abs(self.conscientiousness - other.conscientiousness) <= tolerance and
            abs(self.extraversion - other.extraversion) <= tolerance and
            abs(self.agreeableness - other.agreeableness) <= tolerance and
            abs(self.neuroticism - other.neuroticism) <= tolerance
        )
    
    def get_dominant_traits(self, threshold: int = 60) -> List[str]:
        """Get traits above threshold."""
        traits = []
        if self.openness > threshold:
            traits.append("openness")
        if self.conscientiousness > threshold:
            traits.append("conscientiousness")
        if self.extraversion > threshold:
            traits.append("extraversion")
        if self.agreeableness > threshold:
            traits.append("agreeableness")
        if self.neuroticism > threshold:
            traits.append("neuroticism")
        return traits
    
    def get_weak_traits(self, threshold: int = 40) -> List[str]:
        """Get traits below threshold."""
        traits = []
        if self.openness < threshold:
            traits.append("openness")
        if self.conscientiousness < threshold:
            traits.append("conscientiousness")
        if self.extraversion < threshold:
            traits.append("extraversion")
        if self.agreeableness < threshold:
            traits.append("agreeableness")
        if self.neuroticism < threshold:
            traits.append("neuroticism")
        return traits


class OCEANSampler:
    """
    Sample OCEAN profiles conditioned on demographics.
    
    Uses research-based correlations:
    - Age: +C, -N (older = more conscientious, less neurotic)
    - Sex: Women slightly higher A, N on average
    - Education: +O, +C
    - Occupation: varies by job type
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def sample(
        self,
        age: int,
        sex: str,
        education_level: str,
        occupation: str,
        base_mean: int = 50,
        base_std: int = 15
    ) -> OCEANProfile:
        """
        Sample OCEAN profile conditioned on demographics.
        
        Args:
            age: Person's age
            sex: "Masculin" or "Feminin"
            education_level: Education level
            occupation: Job title
            base_mean: Population mean (default 50)
            base_std: Population std (default 15)
            
        Returns:
            OCEANProfile with demographic adjustments
        """
        # Base sampling
        profile = self._sample_base(base_mean, base_std)
        
        # Apply demographic adjustments
        profile = self._adjust_for_age(profile, age)
        profile = self._adjust_for_sex(profile, sex)
        profile = self._adjust_for_education(profile, education_level)
        profile = self._adjust_for_occupation(profile, occupation)
        
        # Clamp to 0-100
        profile.openness = max(0, min(100, profile.openness))
        profile.conscientiousness = max(0, min(100, profile.conscientiousness))
        profile.extraversion = max(0, min(100, profile.extraversion))
        profile.agreeableness = max(0, min(100, profile.agreeableness))
        profile.neuroticism = max(0, min(100, profile.neuroticism))
        
        return profile
    
    def _sample_base(self, mean: int, std: int) -> OCEANProfile:
        """Sample base profile from normal distribution."""
        def sample_trait():
            # Use bounded normal approximation
            value = self.rng.gauss(mean, std)
            return int(max(0, min(100, value)))
        
        return OCEANProfile(
            openness=sample_trait(),
            conscientiousness=sample_trait(),
            extraversion=sample_trait(),
            agreeableness=sample_trait(),
            neuroticism=sample_trait()
        )
    
    def _adjust_for_age(self, profile: OCEANProfile, age: int) -> OCEANProfile:
        """
        Adjust for age-related personality changes.
        
        Research shows:
        - Conscientiousness increases with age (+0.5 per year)
        - Neuroticism decreases with age (-0.3 per year)
        - Agreeableness increases slightly (+0.2 per year)
        """
        # Normalize age (base at 30)
        age_diff = age - 30
        
        profile.conscientiousness += int(age_diff * 0.5)
        profile.neuroticism -= int(age_diff * 0.3)
        profile.agreeableness += int(age_diff * 0.2)
        
        return profile
    
    def _adjust_for_sex(self, profile: OCEANProfile, sex: str) -> OCEANProfile:
        """
        Small adjustments for sex differences (research-backed but small effects).
        
        Women tend to score slightly higher on:
        - Agreeableness (+3 points)
        - Neuroticism (+2 points)
        """
        if sex == "Feminin":
            profile.agreeableness += 3
            profile.neuroticism += 2
        
        return profile
    
    def _adjust_for_education(self, profile: OCEANProfile, education: str) -> OCEANProfile:
        """
        Adjust for education level.
        
        Higher education associated with:
        - Higher Openness
        - Higher Conscientiousness
        """
        education_effects = {
            "Gimnazial": (-5, 0),
            "Liceal": (0, 0),
            "Profesional/Tehnic": (-2, 2),
            "Postliceal": (3, 3),
            "Superior (Licență/Master)": (8, 5),
            "Doctorat": (12, 8)
        }
        
        o_adjust, c_adjust = education_effects.get(education, (0, 0))
        profile.openness += o_adjust
        profile.conscientiousness += c_adjust
        
        return profile
    
    def _adjust_for_occupation(self, profile: OCEANProfile, occupation: str) -> OCEANProfile:
        """
        Adjust for occupation type.
        
        Different jobs attract different personalities:
        - Creative jobs: +O
        - Detail-oriented: +C
        - Social jobs: +E
        - Caregiving: +A
        - High-stress: +N
        """
        occupation_lower = occupation.lower()
        
        # High Openness occupations
        if any(word in occupation_lower for word in ["artist", "designer", "scriitor", "muzician", "creator"]):
            profile.openness += 10
            profile.extraversion += 3
        
        # High Conscientiousness occupations
        if any(word in occupation_lower for word in ["contabil", "medic", "inginer", "avocat", "judecător", "ofițer"]):
            profile.conscientiousness += 8
            profile.extraversion -= 2
        
        # High Extraversion occupations
        if any(word in occupation_lower for word in ["vânzări", "marketing", "profesor", "manager", "prezentator", "actor"]):
            profile.extraversion += 10
            profile.openness += 3
        
        # High Agreeableness occupations
        if any(word in occupation_lower for word in ["asistent", "îngrijitor", "profesor", "consilier", "psiholog", "asistent social"]):
            profile.agreeableness += 8
            profile.neuroticism -= 3
        
        # High Neuroticism risk occupations (stressful)
        if any(word in occupation_lower for word in ["polițist", "pompier", "medic urgent", "pilot", "jurnalist"]):
            profile.neuroticism += 5
            profile.conscientiousness += 5
        
        return profile


class OCEANBehaviorMapper:
    """
    Map OCEAN traits to behavioral rules and narrative markers.
    
    Used to derive consistent behavioral patterns from personality.
    """
    
    # Risk tolerance rules
    @staticmethod
    def get_risk_tolerance(profile: OCEANProfile) -> str:
        """Determine risk tolerance from OCEAN."""
        # High O + Low N = high risk tolerance
        # Low O + High N = risk averse
        score = profile.openness - profile.neuroticism
        
        if score > 20:
            return "high"  # Entrepreneurial, experimental
        elif score > 0:
            return "moderate"  # Calculated risks
        elif score > -20:
            return "low"  # Conservative
        else:
            return "very_low"  # Risk-averse
    
    # Decision making style
    @staticmethod
    def get_decision_style(profile: OCEANProfile) -> str:
        """Determine decision making style."""
        if profile.conscientiousness > 65 and profile.openness < 45:
            return "deliberative"  # Careful, methodical
        elif profile.openness > 65 and profile.conscientiousness < 50:
            return "intuitive"  # Quick, gut-based
        elif profile.conscientiousness > 65 and profile.openness > 65:
            return "balanced"  # Analytical but open
        elif profile.extraversion > 65:
            return "consultative"  # Asks others
        else:
            return "pragmatic"
    
    # Conflict style
    @staticmethod
    def get_conflict_style(profile: OCEANProfile) -> str:
        """Determine conflict resolution style."""
        if profile.agreeableness > 65 and profile.neuroticism < 45:
            return "accommodating"  # Seeks harmony
        elif profile.agreeableness < 40 and profile.extraversion > 60:
            return "competing"  # Assertive, direct
        elif profile.openness > 60 and profile.agreeableness > 50:
            return "collaborating"  # Win-win
        elif profile.neuroticism > 60:
            return "avoiding"  # Withdraws
        else:
            return "compromising"
    
    # Social energy pattern
    @staticmethod
    def get_social_pattern(profile: OCEANProfile) -> str:
        """Determine social interaction pattern."""
        if profile.extraversion > 70:
            return "gregarious"  # Life of party
        elif profile.extraversion > 50:
            return "outgoing"  # Enjoys groups
        elif profile.extraversion > 30:
            return "selective"  # Small groups
        else:
            return "reserved"  # Prefers solitude
    
    # Novelty seeking
    @staticmethod
    def get_novelty_seeking(profile: OCEANProfile) -> str:
        """Determine preference for novelty vs routine."""
        # High O + High E = high novelty seeking
        # High C + Low O = routine preferring
        score = profile.openness + profile.extraversion - profile.conscientiousness
        
        if score > 80:
            return "explorer"  # Always trying new things
        elif score > 40:
            return "curious"  # Open to new experiences
        elif score > 0:
            return "balanced"  # Mix of both
        else:
            return "traditional"  # Prefers routine
    
    # Generate behavioral contract
    @classmethod
    def generate_behavioral_contract(cls, profile: OCEANProfile) -> Dict[str, Any]:
        """
        Generate a behavioral contract - rules for how this persona behaves.
        
        This is the "structured" layer that guides narrative generation.
        """
        return {
            "ocean_profile": profile.to_dict(),
            "risk_tolerance": cls.get_risk_tolerance(profile),
            "decision_style": cls.get_decision_style(profile),
            "conflict_style": cls.get_conflict_style(profile),
            "social_pattern": cls.get_social_pattern(profile),
            "novelty_seeking": cls.get_novelty_seeking(profile),
            # Derived behavioral markers
            "will_say_no_probability": cls._calc_dissent_probability(profile),
            "complaint_likelihood": cls._calc_complaint_likelihood(profile),
            "planning_horizon": cls._calc_planning_horizon(profile),
        }
    
    @staticmethod
    def _calc_dissent_probability(profile: OCEANProfile) -> float:
        """Calculate probability of saying 'no' or disagreeing."""
        a_norm = profile.agreeableness / 100.0
        e_norm = profile.extraversion / 100.0
        value = 0.05 + 0.35 * (1.0 - a_norm) + 0.10 * e_norm
        return min(0.60, max(0.02, value))
    
    @staticmethod
    def _calc_complaint_likelihood(profile: OCEANProfile) -> str:
        """Determine complaint frequency."""
        # High N + Low A = frequent complaints
        if profile.neuroticism > 65 and profile.agreeableness < 45:
            return "high"
        elif profile.neuroticism > 55 or profile.agreeableness < 40:
            return "moderate"
        elif profile.agreeableness > 65:
            return "low"
        else:
            return "rare"
    
    @staticmethod
    def _calc_planning_horizon(profile: OCEANProfile) -> str:
        """Determine planning time horizon."""
        # High C = long-term planning
        # High O = vision/dreams
        if profile.conscientiousness > 70:
            return "long_term"  # Years ahead
        elif profile.conscientiousness > 55:
            return "medium_term"  # Months ahead
        elif profile.openness > 70:
            return "visionary"  # Big dreams, less detailed
        else:
            return "short_term"  # Day-to-day


class OCEANTextAnalyzer:
    """
    Infer OCEAN traits from text (for validation).
    
    Uses keyword-based analysis as a lightweight approximation.
    For production, could use Receptiviti API or fine-tuned model.
    """
    
    # Keyword mappings (simplified)
    KEYWORDS = {
        "openness": {
            "high": ["creativ", "curios", "imaginație", "artă", "călătorii", "noutate", "experiment", "filozofic", "abstract"],
            "low": ["practic", "tradițional", "concret", "obișnuit", "sigur", "predictibil"]
        },
        "conscientiousness": {
            "high": ["organizat", "disciplinat", "planific", "precis", "responsabil", "meticulos", "punctual", "structurat"],
            "low": ["spontan", "relaxat", "flexibil", "improvizează", "dezorganizat"]
        },
        "extraversion": {
            "high": ["sociabil", "extrovertit", "energic", "vorbăreț", "petreceri", "grupuri", "entuziast", "asertiv"],
            "low": ["introvertit", "rezervat", "solitar", "liniștit", "introspectiv", "tăcut"]
        },
        "agreeableness": {
            "high": ["amabil", "cooperant", "empatic", "înțelegător", "generos", "ajută", "armonios", "diplomatic"],
            "low": ["competitiv", "direct", "critic", "sceptic", "dur", "confruntațional"]
        },
        "neuroticism": {
            "high": ["anxios", "stresat", "îngrijorat", "emoțional", "sensibil", "nervos", "fricos", "temător"],
            "low": ["calm", "stabil", "relaxat", "încrezător", "rezilient", "echilibrat", "liniștit"]
        }
    }

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text
    
    def analyze(self, text: str) -> OCEANProfile:
        """
        Infer OCEAN profile from text.
        
        Returns inferred profile with confidence score.
        """
        normalized_text = self._normalize_text(text)
        tokens = re.findall(r"\b\w+\b", normalized_text)
        token_set = set(tokens)

        if not tokens:
            return OCEANProfile(
                openness=50,
                conscientiousness=50,
                extraversion=50,
                agreeableness=50,
                neuroticism=50,
                source="inferred",
                confidence=0.0
            )
        
        scores = {}
        total_hits = 0
        for trait, keywords in self.KEYWORDS.items():
            high_words = {self._normalize_text(word) for word in keywords["high"]}
            low_words = {self._normalize_text(word) for word in keywords["low"]}

            high_hits = {word for word in high_words if word in token_set}
            low_hits = {word for word in low_words if word in token_set}

            trait_hits = len(high_hits) + len(low_hits)
            total_hits += trait_hits

            delta = 5 * (len(high_hits) - len(low_hits))
            delta = max(-20, min(20, delta))
            if trait_hits < 2:
                delta = max(-10, min(10, delta))

            scores[trait] = int(max(0, min(100, 50 + delta)))
        
        # Confidence based on keyword density
        total_words = len(tokens)
        confidence = min(1.0, total_hits / max(1.0, total_words * 0.1))  # Expect ~10% keyword density
        
        return OCEANProfile(
            openness=scores["openness"],
            conscientiousness=scores["conscientiousness"],
            extraversion=scores["extraversion"],
            agreeableness=scores["agreeableness"],
            neuroticism=scores["neuroticism"],
            source="inferred",
            confidence=confidence
        )
    
    def calculate_deviation(
        self, 
        target: OCEANProfile, 
        text: str,
        tolerance: int = 10
    ) -> Tuple[bool, Dict[str, int], OCEANProfile]:
        """
        Check if text matches target OCEAN profile.
        
        Returns:
            (is_within_tolerance, per_trait_deviations, inferred_profile)
        """
        inferred = self.analyze(text)
        
        deviations = {
            "openness": inferred.openness - target.openness,
            "conscientiousness": inferred.conscientiousness - target.conscientiousness,
            "extraversion": inferred.extraversion - target.extraversion,
            "agreeableness": inferred.agreeableness - target.agreeableness,
            "neuroticism": inferred.neuroticism - target.neuroticism
        }
        
        is_within = all(abs(d) <= tolerance for d in deviations.values())
        
        return is_within, deviations, inferred


def generate_ocean_guided_prompt(
    ocean_profile: OCEANProfile,
    behavioral_contract: Dict[str, Any]
) -> str:
    """
    Generate a prompt modifier based on OCEAN profile.
    
    This guides the LLM to generate consistent personality expression.
    """
    traits_desc = []
    
    # Describe each trait
    if ocean_profile.openness > 60:
        traits_desc.append("curios și deschis la experiențe noi")
    elif ocean_profile.openness < 40:
        traits_desc.append("practic și preferă rutina")
    
    if ocean_profile.conscientiousness > 60:
        traits_desc.append("organizat și responsabil")
    elif ocean_profile.conscientiousness < 40:
        traits_desc.append("spontan și flexibil")
    
    if ocean_profile.extraversion > 60:
        traits_desc.append("sociabil și energic")
    elif ocean_profile.extraversion < 40:
        traits_desc.append("rezervat și introspectiv")
    
    if ocean_profile.agreeableness > 60:
        traits_desc.append("amabil și cooperant")
    elif ocean_profile.agreeableness < 40:
        traits_desc.append("direct și competitiv")
    
    if ocean_profile.neuroticism > 60:
        traits_desc.append("emotional și precaut")
    elif ocean_profile.neuroticism < 40:
        traits_desc.append("calm și stabil emoțional")
    
    # Add behavioral specifics
    risk = behavioral_contract["risk_tolerance"]
    decision = behavioral_contract["decision_style"]
    conflict = behavioral_contract["conflict_style"]
    social = behavioral_contract["social_pattern"]
    novelty = behavioral_contract["novelty_seeking"]

    if traits_desc:
        personality_line = f"Persoana este {', '.join(traits_desc)}."
    else:
        personality_line = "Persoana are un profil echilibrat și ia decizii pragmatic."
    
    prompt = f"""PERSONALITATE (OCEAN Profile):
- Openness: {ocean_profile.openness}/100
- Conscientiousness: {ocean_profile.conscientiousness}/100  
- Extraversion: {ocean_profile.extraversion}/100
- Agreeableness: {ocean_profile.agreeableness}/100
- Neuroticism: {ocean_profile.neuroticism}/100

    DESCRIERE PERSONALITATE:
    {personality_line}

COMPORTAMENT:
- Toleranță la risc: {risk}
- Stil decizional: {decision}
- Stil conflict: {conflict}
- Pattern social: {social}
- Preferință noutate: {novelty}
- Probabilitate de a spune "nu": {behavioral_contract['will_say_no_probability']:.0%}
- Frecvență plângeri: {behavioral_contract['complaint_likelihood']}
- Orizont planificare: {behavioral_contract['planning_horizon']}

INSTRUCȚIUNI PENTRU GENERARE:
Generează descrieri care reflectă ACEASTĂ personalitate specifică.
Evită clișee pozitive - această persoană poate fi critică, anxioasă, sau competitivă dacă profilul OCEAN indică asta.
Asigură coerența între toate descrierile - aceeași persoană în contexte diferite.
"""
    
    return prompt


# Convenience functions
def sample_ocean_profile(
    age: int,
    sex: str,
    education: str,
    occupation: str
) -> OCEANProfile:
    """Sample OCEAN profile conditioned on demographics."""
    sampler = OCEANSampler()
    return sampler.sample(age, sex, education, occupation)


def generate_behavioral_contract(profile: OCEANProfile) -> Dict[str, Any]:
    """Generate behavioral rules from OCEAN profile."""
    return OCEANBehaviorMapper.generate_behavioral_contract(profile)
