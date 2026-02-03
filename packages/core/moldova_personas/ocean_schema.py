"""
Reference OCEAN schema with t_score, label, description.

Matches the reference dataset's structured personality representation.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OCEANTrait:
    """
    Single OCEAN trait in reference format.
    
    Attributes:
        t_score: T-score (0-100, population-normalized)
        label: Categorical label (e.g., "high", "average", "low")
        description: Human-readable description of trait expression
    """
    t_score: int = 50
    label: str = "average"
    description: str = ""
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "t_score": self.t_score,
            "label": self.label,
            "description": self.description
        }
    
    @classmethod
    def from_raw_score(cls, score: int, trait_name: str) -> "OCEANTrait":
        """Create from raw 0-100 score with auto-generated label and description."""
        # Determine label
        if score >= 70:
            label = "high"
        elif score >= 40:
            label = "average"
        else:
            label = "low"
        
        # Generate description based on trait and label
        description = cls._generate_description(trait_name, score, label)
        
        return cls(t_score=score, label=label, description=description)
    
    @staticmethod
    def _generate_description(trait: str, score: int, label: str) -> str:
        """Generate human-readable trait description."""
        descriptions = {
            "openness": {
                "high": "Curios, creativ, deschis la experiențe noi și idei abstracte",
                "average": "Echilibrat între tradiție și noutate",
                "low": "Practic, preferă rutina și concretețea"
            },
            "conscientiousness": {
                "high": "Organizat, disciplinat, planificat, responsabil",
                "average": "Moderat organizat, flexibil când e necesar",
                "low": "Spontan, preferă flexibilitatea și improvizația"
            },
            "extraversion": {
                "high": "Sociabil, energic, asertiv, caută stimulare externă",
                "average": "Selectiv social, echilibrat între solitudine și companie",
                "low": "Rezervat, introspectiv, preferă activități solitare"
            },
            "agreeableness": {
                "high": "Cooperant, empatic, altruist, evită conflictele",
                "average": "Direct dar respectuos, echilibrat",
                "low": "Competitiv, direct, sceptic, asertiv"
            },
            "neuroticism": {
                "high": "Emoțional, anxios, sensibil la stres, precaut",
                "average": "Reactivitate emoțională moderată",
                "low": "Calm, stabil emoțional, rezilient, încrezător"
            }
        }
        
        return descriptions.get(trait, {}).get(label, f"Nivel {label} de {trait}")


@dataclass
class OCEANProfileSchema:
    """
    Complete OCEAN profile in reference schema format.
    
    Each trait is a structured object with t_score, label, description.
    """
    openness: OCEANTrait
    conscientiousness: OCEANTrait
    extraversion: OCEANTrait
    agreeableness: OCEANTrait
    neuroticism: OCEANTrait
    
    # Metadata
    source: str = "sampled"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "openness": self.openness.to_dict(),
            "conscientiousness": self.conscientiousness.to_dict(),
            "extraversion": self.extraversion.to_dict(),
            "agreeableness": self.agreeableness.to_dict(),
            "neuroticism": self.neuroticism.to_dict(),
            "source": self.source,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_raw_scores(
        cls,
        openness: int,
        conscientiousness: int,
        extraversion: int,
        agreeableness: int,
        neuroticism: int,
        source: str = "sampled",
        confidence: float = 1.0
    ) -> "OCEANProfileSchema":
        """Create from raw 0-100 scores."""
        return cls(
            openness=OCEANTrait.from_raw_score(openness, "openness"),
            conscientiousness=OCEANTrait.from_raw_score(conscientiousness, "conscientiousness"),
            extraversion=OCEANTrait.from_raw_score(extraversion, "extraversion"),
            agreeableness=OCEANTrait.from_raw_score(agreeableness, "agreeableness"),
            neuroticism=OCEANTrait.from_raw_score(neuroticism, "neuroticism"),
            source=source,
            confidence=confidence
        )
    
    def get_trait_summary(self) -> str:
        """Get summary of dominant traits."""
        traits = []
        
        if self.openness.t_score >= 70:
            traits.append("high openness")
        elif self.openness.t_score <= 30:
            traits.append("low openness")
            
        if self.conscientiousness.t_score >= 70:
            traits.append("high conscientiousness")
        elif self.conscientiousness.t_score <= 30:
            traits.append("low conscientiousness")
            
        if self.extraversion.t_score >= 70:
            traits.append("high extraversion")
        elif self.extraversion.t_score <= 30:
            traits.append("low extraversion")
            
        if self.agreeableness.t_score >= 70:
            traits.append("high agreeableness")
        elif self.agreeableness.t_score <= 30:
            traits.append("low agreeableness")
            
        if self.neuroticism.t_score >= 70:
            traits.append("high neuroticism")
        elif self.neuroticism.t_score <= 30:
            traits.append("low neuroticism")
        
        return ", ".join(traits) if traits else "balanced profile"


def convert_to_reference_schema(raw_scores: Dict[str, int]) -> Dict[str, any]:
    """
    Convert raw OCEAN scores to reference schema.
    
    Args:
        raw_scores: Dict with keys openness, conscientiousness, extraversion, 
                   agreeableness, neuroticism (values 0-100)
    
    Returns:
        Dict in reference format with t_score, label, description for each trait
    """
    profile = OCEANProfileSchema.from_raw_scores(
        openness=raw_scores.get("openness", 50),
        conscientiousness=raw_scores.get("conscientiousness", 50),
        extraversion=raw_scores.get("extraversion", 50),
        agreeableness=raw_scores.get("agreeableness", 50),
        neuroticism=raw_scores.get("neuroticism", 50)
    )
    
    return profile.to_dict()
