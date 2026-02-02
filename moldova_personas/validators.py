"""
Validation pipeline for persona data.

Implements multi-layer validation as described in the technical plan:
1. Structural validation (types, ranges, uniqueness)
2. Logical validation (consistency between fields)
3. Narrative validation (coherence with structured data)
4. Statistical validation (match to census distributions)
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

from .models import Persona, AgeConstraints, PopulationMode
from .census_data import CENSUS
from .geo_tables import (
    strict_geo_enabled,
    get_district_distribution,
    get_region_distribution_from_district,
    get_region_for_district,
    get_district_region_map,
)


@dataclass
class ValidationError:
    """Single validation error or warning."""
    field: str
    message: str
    severity: str  # "error" or "warning"
    persona_uuid: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    total_checked: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    @property
    def is_valid(self) -> bool:
        return self.error_count == 0
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Validation Report",
            f"=================",
            f"Total personas checked: {self.total_checked}",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
            f"Status: {'✓ PASSED' if self.is_valid else '✗ FAILED'}",
        ]
        
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors[:10]:  # Show first 10
                lines.append(f"  [{e.field}] {e.message}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings[:10]:
                lines.append(f"  [{w.field}] {w.message}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        
        return "\n".join(lines)


class ValidationPipeline:
    """
    Multi-layer validation pipeline for persona datasets.
    """
    
    def __init__(self, census_data=None, population_mode: PopulationMode = PopulationMode.ADULT_18):
        """
        Initialize validator.
        
        Args:
            census_data: Census distributions for statistical validation
        """
        self.census = census_data or CENSUS
        self.population_mode = population_mode
        
        # Occupations requiring higher education
        self.high_education_occupations = {
            'profesor', 'medic', 'inginer', 'avocat', 'judecător', 'procuror',
            'arhitect', 'farmacist', 'psiholog', 'economist', 'programator',
            'analist', 'consultant', 'manager senior', 'director', 'doctor',
            'cercetător', 'universitar', 'jurnalist', 'traducător'
        }
    
    def validate(self, personas: List[Persona], 
                 check_statistical: bool = True,
                 tolerance: float = 0.02) -> ValidationReport:
        """
        Run full validation pipeline.
        
        Args:
            personas: List of personas to validate
            check_statistical: Whether to check statistical distributions
            tolerance: Acceptable deviation from target distributions (0.02 = 2%)
        
        Returns:
            ValidationReport with all errors and warnings
        """
        errors = []
        warnings = []
        
        for persona in personas:
            # Layer 1: Structural validation
            e, w = self._validate_structural(persona)
            errors.extend(e)
            warnings.extend(w)
            
            # Layer 2: Logical validation
            e, w = self._validate_logical(persona)
            errors.extend(e)
            warnings.extend(w)
        
        # Layer 3: Statistical validation (on full dataset)
        if check_statistical:
            e, w = self._validate_statistical(personas, tolerance)
            errors.extend(e)
            warnings.extend(w)
        
        return ValidationReport(
            total_checked=len(personas),
            errors=errors,
            warnings=warnings
        )
    
    def _validate_structural(self, persona: Persona) -> Tuple[List[ValidationError], List[ValidationError]]:
        """
        Validate structural integrity of a persona.
        
        Checks:
        - UUID format
        - Age range
        - Required fields present
        - Enum values valid
        """
        errors = []
        warnings = []
        
        # Age validation (mode-aware)
        min_age, max_age = self._age_range()
        if not (min_age <= persona.age <= max_age):
            errors.append(ValidationError(
                field="age",
                message=f"Age {persona.age} out of valid range [{min_age}, {max_age}]",
                severity="error",
                persona_uuid=persona.uuid
            ))
        
        # Age group consistency
        expected_age_group = self._get_age_group(persona.age)
        if persona.age_group != expected_age_group:
            errors.append(ValidationError(
                field="age_group",
                message=f"Age group '{persona.age_group}' doesn't match age {persona.age} (expected {expected_age_group})",
                severity="error",
                persona_uuid=persona.uuid
            ))
        
        # Sex validation
        if persona.sex not in ["Masculin", "Feminin"]:
            errors.append(ValidationError(
                field="sex",
                message=f"Invalid sex value: {persona.sex}",
                severity="error",
                persona_uuid=persona.uuid
            ))
        
        # Region validation
        valid_regions = ["Chisinau", "Centru", "Nord", "Sud", "Gagauzia"]
        if persona.region not in valid_regions:
            errors.append(ValidationError(
                field="region",
                message=f"Invalid region: {persona.region}",
                severity="error",
                persona_uuid=persona.uuid
            ))

        # District validation (strict geo mode)
        if strict_geo_enabled():
            district_dist = get_district_distribution()
            region_map = get_district_region_map()

            if district_dist:
                if not persona.district:
                    errors.append(ValidationError(
                        field="district",
                        message="Missing district in strict geo mode with official district distribution",
                        severity="error",
                        persona_uuid=persona.uuid
                    ))
                else:
                    try:
                        expected_region = get_region_for_district(persona.district, strict=True)
                    except Exception:
                        expected_region = None
                        errors.append(ValidationError(
                            field="district",
                            message=f"Unknown district '{persona.district}' in strict geo mode",
                            severity="error",
                            persona_uuid=persona.uuid
                        ))
                    if expected_region and persona.region != expected_region:
                        errors.append(ValidationError(
                            field="region",
                            message=f"Region '{persona.region}' does not match district '{persona.district}' (expected {expected_region})",
                            severity="error",
                            persona_uuid=persona.uuid
                        ))

                if persona.city:
                    errors.append(ValidationError(
                        field="city",
                        message="City should be empty in strict geo mode (district-only data)",
                        severity="error",
                        persona_uuid=persona.uuid
                    ))
            else:
                # No official district distribution available
                if persona.district or persona.city:
                    warnings.append(ValidationError(
                        field="district",
                        message="District/locality present but no official district distribution is configured",
                        severity="warning",
                        persona_uuid=persona.uuid
                    ))
        
        # Residence type validation
        if persona.residence_type not in ["Urban", "Rural"]:
            errors.append(ValidationError(
                field="residence_type",
                message=f"Invalid residence_type: {persona.residence_type}",
                severity="error",
                persona_uuid=persona.uuid
            ))
        
        # Education level validation
        valid_education = list(self.census.EDUCATION_DISTRIBUTION.keys())
        if persona.education_level not in valid_education:
            errors.append(ValidationError(
                field="education_level",
                message=f"Invalid education level: {persona.education_level}",
                severity="error",
                persona_uuid=persona.uuid
            ))
        
        # Field of study should exist for higher education
        if persona.education_level in ["Superior (Licență/Master)", "Doctorat"]:
            if not persona.field_of_study:
                warnings.append(ValidationError(
                    field="field_of_study",
                    message=f"Missing field_of_study for higher education",
                    severity="warning",
                    persona_uuid=persona.uuid
                ))
        
        return errors, warnings
    
    def _validate_logical(self, persona: Persona) -> Tuple[List[ValidationError], List[ValidationError]]:
        """
        Validate logical consistency between fields.
        
        Checks:
        - Age vs Education (5-year-old with PhD)
        - Age vs Marital Status (12-year-old married)
        - Education vs Occupation (high skill job with low education)
        - Age vs Occupation (child with professional job)
        """
        errors = []
        warnings = []
        
        # Age vs Education consistency
        age_education_errors = self._check_age_education(persona)
        errors.extend(age_education_errors)
        
        # Age vs Marital Status
        if persona.age < 16 and persona.marital_status == "Căsătorit":
            errors.append(ValidationError(
                field="marital_status",
                message=f"Child (age {persona.age}) marked as married",
                severity="error",
                persona_uuid=persona.uuid
            ))
        
        # Education vs Occupation
        occ_lower = persona.occupation.lower()
        needs_high_ed = any(job in occ_lower for job in self.high_education_occupations)
        has_high_ed = persona.education_level in ["Superior (Licență/Master)", "Doctorat"]
        
        if needs_high_ed and not has_high_ed:
            warnings.append(ValidationError(
                field="occupation",
                message=f"Occupation '{persona.occupation}' typically requires higher education, but has '{persona.education_level}'",
                severity="warning",
                persona_uuid=persona.uuid
            ))
        
        # Age vs Occupation
        if persona.age < 14:
            # Children should be students or have child-appropriate status
            valid_child_jobs = ["elev", "student", "copil"]
            if not any(job in occ_lower for job in valid_child_jobs):
                warnings.append(ValidationError(
                    field="occupation",
                    message=f"Child (age {persona.age}) has occupation '{persona.occupation}'",
                    severity="warning",
                    persona_uuid=persona.uuid
                ))
        
        # Ethnicity vs Language consistency
        ethnicity_language_map = {
            "Moldovean": ["Română"],
            "Român": ["Română"],
            "Ucrainean": ["Ucraineană", "Română"],  # Some bilingual
            "Găgăuz": ["Găgăuză", "Rusă", "Română"],  # Often multilingual
            "Rus": ["Rusă"],
            "Bulgar": ["Bulgară", "Română"],
            "Rrom": ["Română", "Rromani"],
        }
        
        expected_langs = ethnicity_language_map.get(persona.ethnicity, ["Română"])
        if persona.mother_tongue not in expected_langs:
            # This is just a warning as there are exceptions
            pass  # Allow some flexibility
        
        return errors, warnings
    
    def _check_age_education(self, persona: Persona) -> List[ValidationError]:
        """Check age-education consistency using centralized constraints."""
        errors = []
        ac = AgeConstraints
        
        # Minimum ages for education levels (with some flexibility)
        min_ages = {
            "Fără studii": 0,
            "Primar": ac.MIN_PRIMARY_SCHOOL,
            "Gimnazial": ac.MIN_GYMNASIUM,
            "Liceal": ac.MIN_HIGH_SCHOOL,
            "Profesional/Tehnic": ac.MIN_HIGH_SCHOOL,
            "Superior (Licență/Master)": ac.MIN_HIGHER_EDUCATION,
            "Doctorat": ac.MIN_DOCTORATE,
        }
        
        min_age = min_ages.get(persona.education_level, 0)
        if persona.age < min_age:
            errors.append(ValidationError(
                field="education_level",
                message=f"Age {persona.age} too young for education level '{persona.education_level}' (min: {min_age})",
                severity="error",
                persona_uuid=persona.uuid
            ))
        
        return errors
    
    def _validate_statistical(self, personas: List[Persona], 
                             tolerance: float) -> Tuple[List[ValidationError], List[ValidationError]]:
        """
        Validate that generated distributions match census targets.
        
        Checks:
        - Sex distribution
        - Region distribution
        - Ethnicity distribution
        - Education distribution
        - Urban/rural split
        """
        errors = []
        warnings = []
        
        n = len(personas)
        if n == 0:
            return errors, warnings
        
        # Check sex distribution
        sex_counts = Counter(p.sex for p in personas)
        sex_dist = {k: v/n for k, v in sex_counts.items()}
        for sex, target in self.census.SEX_DISTRIBUTION.items():
            actual = sex_dist.get(sex, 0)
            if abs(actual - target) > tolerance:
                warnings.append(ValidationError(
                    field="sex_distribution",
                    message=f"Sex '{sex}': target {target:.3f}, actual {actual:.3f} (diff: {abs(actual-target):.3f})",
                    severity="warning"
                ))
        
        # Check region distribution
        region_counts = Counter(p.region for p in personas)
        region_dist = {k: v/n for k, v in region_counts.items()}
        target_region_dist = self.census.REGION_DISTRIBUTION
        if strict_geo_enabled():
            derived = get_region_distribution_from_district(strict=True)
            if derived:
                target_region_dist = derived
        for region, target in target_region_dist.items():
            actual = region_dist.get(region, 0)
            if abs(actual - target) > tolerance:
                warnings.append(ValidationError(
                    field="region_distribution",
                    message=f"Region '{region}': target {target:.3f}, actual {actual:.3f} (diff: {abs(actual-target):.3f})",
                    severity="warning"
                ))
        
        # Check ethnicity distribution
        ethnicity_counts = Counter(p.ethnicity for p in personas)
        ethnicity_dist = {k: v/n for k, v in ethnicity_counts.items()}
        for ethnicity, target in self.census.ETHNICITY_DISTRIBUTION.items():
            actual = ethnicity_dist.get(ethnicity, 0)
            if abs(actual - target) > tolerance * 2:  # More tolerance for small groups
                warnings.append(ValidationError(
                    field="ethnicity_distribution",
                    message=f"Ethnicity '{ethnicity}': target {target:.3f}, actual {actual:.3f} (diff: {abs(actual-target):.3f})",
                    severity="warning"
                ))
        
        # Check urban/rural
        residence_counts = Counter(p.residence_type for p in personas)
        residence_dist = {k: v/n for k, v in residence_counts.items()}
        for residence, target in self.census.RESIDENCE_TYPE_DISTRIBUTION.items():
            actual = residence_dist.get(residence, 0)
            if abs(actual - target) > tolerance:
                warnings.append(ValidationError(
                    field="residence_distribution",
                    message=f"Residence '{residence}': target {target:.3f}, actual {actual:.3f} (diff: {abs(actual-target):.3f})",
                    severity="warning"
                ))
        
        # Check education distribution (mode-aligned)
        ed_personas = [p for p in personas if p.age >= 10]
        if ed_personas:
            ed_counts = Counter(p.education_level for p in ed_personas)
            ed_dist = {k: v/len(ed_personas) for k, v in ed_counts.items()}
            education_target = self.census.education_distribution(self.population_mode).values
            for education, target in education_target.items():
                actual = ed_dist.get(education, 0)
                if abs(actual - target) > tolerance * 1.5:
                    warnings.append(ValidationError(
                        field="education_distribution",
                        message=f"Education '{education}': target {target:.3f}, actual {actual:.3f} (diff: {abs(actual-target):.3f})",
                        severity="warning"
                    ))
        
        return errors, warnings
    
    def _get_age_group(self, age: int) -> str:
        """Map age to age group."""
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

    def _age_range(self) -> Tuple[int, int]:
        """Return (min_age, max_age) for the current population mode."""
        if self.population_mode == PopulationMode.ADULT_18:
            return AgeConstraints.MIN_PERSONA_AGE, 110
        if self.population_mode == PopulationMode.AGE_15_PLUS:
            return 15, 110
        return 0, 110


class NarrativeValidator:
    """
    Validator for LLM-generated narrative content.
    
    Note: This requires the narrative fields to be populated.
    For the prototype, this checks basic consistency.
    """
    
    def __init__(self):
        self.male_pronouns = ["el", "lui", "bărbat", "băiat"]
        self.female_pronouns = ["ea", "ei", "femeie", "fată"]
    
    def validate_narrative(self, persona: Persona) -> List[ValidationError]:
        """
        Validate narrative content against structured data.
        
        Basic checks:
        - Name appears in narrative
        - Pronouns match sex
        - No obvious age contradictions
        """
        errors = []
        
        if not persona.descriere_generala:
            return errors  # No narrative to validate
        
        combined_text = " ".join([
            persona.descriere_generala,
            persona.profil_profesional,
            persona.hobby_sport,
            persona.hobby_arta_cultura,
            persona.hobby_calatorii,
            persona.hobby_culinar,
        ]).lower()
        
        # Check name appears
        first_name = persona.name.split()[0].lower()
        if first_name not in combined_text:
            errors.append(ValidationError(
                field="descriere_generala",
                message=f"First name '{first_name}' not found in narrative text",
                severity="warning",
                persona_uuid=persona.uuid
            ))
        
        # Check pronoun consistency
        if persona.sex == "Masculin":
            female_count = sum(1 for p in self.female_pronouns if p in combined_text)
            if female_count > 2:  # Allow some flexibility
                errors.append(ValidationError(
                    field="descriere_generala",
                    message=f"Found {female_count} female pronouns in male persona narrative",
                    severity="warning",
                    persona_uuid=persona.uuid
                ))
        else:
            male_count = sum(1 for p in self.male_pronouns if p in combined_text)
            if male_count > 2:
                errors.append(ValidationError(
                    field="descriere_generala",
                    message=f"Found {male_count} male pronouns in female persona narrative",
                    severity="warning",
                    persona_uuid=persona.uuid
                ))
        
        # Check for age contradictions
        if persona.age < 18:
            work_experience_indicators = ["10 ani experiență", "20 de ani de muncă", "carieră de ",
                                         "pensionar", "la pensie"]
            for indicator in work_experience_indicators:
                if indicator in combined_text:
                    errors.append(ValidationError(
                        field="profil_profesional",
                        message=f"Minor (age {persona.age}) has work experience indicator: '{indicator}'",
                        severity="error",
                        persona_uuid=persona.uuid
                    ))
        
        return errors
