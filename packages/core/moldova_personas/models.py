"""
Pydantic models for persona data structure.

Defines the schema for both structured fields and narrative fields,
ensuring type safety and validation.
"""

from typing import Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict


class AgeConstraints:
    """Centralized age constraints for consistency across the pipeline."""
    MIN_PERSONA_AGE = 18           # Minimum age for generated personas
    MIN_PRIMARY_SCHOOL = 6         # Age to start primary school
    MIN_GYMNASIUM = 10             # Age for gymnasium
    MIN_HIGH_SCHOOL = 14           # Age for high school/vocational
    MIN_HIGHER_EDUCATION = 19      # Age for university (licență/master)
    MIN_DOCTORATE = 27             # Age for PhD (typical completion)
    RETIREMENT_AGE = 63            # Standard retirement age
    MAX_REALISTIC_AGE = 90         # Cap for realistic age generation


class Persona(BaseModel):
    """
    Complete persona model for synthetic Moldovan adults (18+).
    
    This schema represents working-age and retirement-age adults only.
    Age range: 18-90 (configurable via AgeConstraints)
    
    Contains both structured demographic fields and narrative text fields.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Ion Ciobanu",
                "sex": "Masculin",
                "age": 52,
                "age_group": "45-54",
                "ethnicity": "Moldovean",
                "mother_tongue": "Română",
                "religion": "Ortodox",
                "marital_status": "Căsătorit",
                "education_level": "Profesional/Tehnic",
                "occupation": "Șofer de autobuz",
                "city": "Ungheni",
                "district": "Ungheni",
                "region": "Centru",
                "residence_type": "Urban",
                "country": "Moldova",
            }
        }
    )
    
    # =========================================================================
    # CORE IDENTIFIERS
    # =========================================================================
    uuid: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    
    # =========================================================================
    # STRUCTURED DEMOGRAPHIC FIELDS
    # =========================================================================
    name: str = Field(..., description="Full name (first + last)")
    sex: str = Field(..., pattern="^(Masculin|Feminin)$")
    age: int = Field(
        ..., 
        ge=AgeConstraints.MIN_PERSONA_AGE, 
        le=AgeConstraints.MAX_REALISTIC_AGE,
        description=f"Age in years ({AgeConstraints.MIN_PERSONA_AGE}-{AgeConstraints.MAX_REALISTIC_AGE})"
    )
    
    # Derived age group for sampling/validation (adult-only: 18-24 mapped to 15-24, 25-34, etc.)
    age_group: str = Field(..., description="Age category (15-24, 25-34, 35-44, 45-54, 55-64, 65+)")
    
    ethnicity: str = Field(..., description="Ethnic self-identification")
    mother_tongue: str = Field(..., description="Native language")
    religion: str = Field(..., description="Religious affiliation")
    
    marital_status: str = Field(..., description="Marital status")
    education_level: str = Field(..., description="Highest education level (ISCED)")
    field_of_study: Optional[str] = Field(None, description="Field of study for higher education")
    
    occupation: str = Field(..., description="Current occupation")
    occupation_sector: Optional[str] = Field(None, description="Economic sector")
    employment_status: str = Field(
        default="employed",
        description="Employment status (employed, unemployed, student, retired, homemaker, other_inactive)"
    )
    
    # Geographic fields
    city: str = Field(..., description="City, town, or village name")
    district: str = Field(..., description="District/Raion")
    region: str = Field(..., pattern="^(Chisinau|Centru|Nord|Sud|Gagauzia)$")
    residence_type: str = Field(..., pattern="^(Urban|Rural)$")
    country: str = Field(default="Moldova", description="Country of residence")
    
    # =========================================================================
    # NARRATIVE FIELDS (LLM-generated)
    # =========================================================================
    # Persona variants (diagram: LLM B outputs)
    persona: str = Field(
        default="",
        description="Overall persona essence (short variant)"
    )
    professional_persona: str = Field(
        default="",
        description="Professional persona variant"
    )
    sports_persona: str = Field(
        default="",
        description="Sports persona variant"
    )
    arts_persona: str = Field(
        default="",
        description="Arts/cultural persona variant"
    )
    travel_persona: str = Field(
        default="",
        description="Travel persona variant"
    )
    culinary_persona: str = Field(
        default="",
        description="Culinary persona variant"
    )

    # Context fields (diagram: LLM A outputs)
    cultural_background: str = Field(
        default="", 
        description="Ethnic and cultural context description"
    )
    skills_and_expertise: str = Field(
        default="",
        description="Skills and expertise narrative"
    )
    hobbies_and_interests: str = Field(
        default="",
        description="Hobbies and interests narrative"
    )
    descriere_generala: str = Field(
        default="",
        description="General personality description in Romanian"
    )
    profil_profesional: str = Field(
        default="",
        description="Professional life and career description in Romanian"
    )
    hobby_sport: str = Field(
        default="",
        description="Sports and physical activities in Romanian"
    )
    hobby_arta_cultura: str = Field(
        default="",
        description="Cultural and artistic interests in Romanian"
    )
    hobby_calatorii: str = Field(
        default="",
        description="Travel preferences in Romanian"
    )
    hobby_culinar: str = Field(
        default="",
        description="Culinary habits in Romanian"
    )
    career_goals_and_ambitions: str = Field(
        default="",
        description="Career aspirations and future plans in Romanian"
    )
    persona_summary: str = Field(
        default="",
        description="Brief one-liner summary of the persona"
    )
    
    # =========================================================================
    # STRUCTURED DERIVED FIELDS
    # =========================================================================
    skills_and_expertise_list: List[str] = Field(
        default_factory=list,
        description="Professional skills extracted from narrative"
    )
    hobbies_and_interests_list: List[str] = Field(
        default_factory=list,
        description="Hobbies extracted from narrative"
    )

    # =========================================================================
    # OCEAN PERSONALITY (0-100) - populated by narrative pipeline
    # =========================================================================
    ocean_openness: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="OCEAN openness score (0-100)"
    )
    ocean_conscientiousness: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="OCEAN conscientiousness score (0-100)"
    )
    ocean_extraversion: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="OCEAN extraversion score (0-100)"
    )
    ocean_agreeableness: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="OCEAN agreeableness score (0-100)"
    )
    ocean_neuroticism: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="OCEAN neuroticism score (0-100)"
    )
    ocean_source: Optional[str] = Field(
        default=None,
        description="OCEAN source (sampled/estimated)"
    )
    ocean_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="OCEAN confidence (0-1)"
    )
    ocean_deviation_score: Optional[float] = Field(
        default=None,
        description="Max deviation between target and inferred OCEAN in narrative"
    )
    rewrite_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of narrative rewrites for OCEAN consistency"
    )
    
    # =========================================================================
    # METADATA & TRUST VECTOR
    # =========================================================================
    generation_timestamp: Optional[str] = Field(None, description="ISO timestamp of generation")
    census_reference_year: int = Field(default=2024, description="Reference census year")
    
    # Narrative generation status for auditability
    narrative_status: str = Field(
        default="pending",
        pattern="^(pending|generated|mock|failed)$",
        description="Status of narrative field generation: pending=not yet, generated=LLM success, mock=empty fallback, failed=error"
    )
    
    # Provenance tracking for data sources (populated at dataset level, copied here)
    field_provenance: Optional[Dict[str, str]] = Field(
        default=None,
        description="Map of field names to data source (PXWEB_DIRECT, CENSUS_HARDCODED, IPF_DERIVED, ESTIMATED, LLM_GENERATED)"
    )
    



class PersonaStatistics(BaseModel):
    """Summary statistics for a generated persona dataset."""
    
    total_count: int
    
    # Demographic distributions
    sex_distribution: dict
    age_distribution: dict
    region_distribution: dict
    ethnicity_distribution: dict
    education_distribution: dict
    marital_status_distribution: dict
    urban_rural_distribution: dict
    employment_status_distribution: Optional[dict] = None
    
    # Validation results
    validation_errors: int
    validation_warnings: int
    
    # Generation metadata
    generation_duration_seconds: Optional[float] = None
    llm_tokens_used: Optional[int] = None


def get_age_group(age: int) -> str:
    """
    Map age to age group category.
    
    Args:
        age: Age in years
    
    Returns:
        Age group string
    """
    if age < 15:
        return "0-14"
    elif age < 25:
        return "15-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"
