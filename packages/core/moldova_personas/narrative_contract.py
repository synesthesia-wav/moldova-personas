"""
Narrative Contract Enforcement

Prevents format drift in LLM-generated narratives by:
1. Fixed persona regression tests (detect breaking changes)
2. JSON schema validation (when structured output is available)
3. Response format validation (section markers, required fields)
4. Golden narrative fixtures for comparison

This ensures that narrative generation remains stable over time
and provides early warning when LLM responses change format.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from .models import Persona
from .prompts import (
    PROMPT_VERSION,
    parse_narrative_response,
    parse_json_narrative_response_strict,
    validate_narrative_against_persona,
)


# Version of the narrative contract
NARRATIVE_CONTRACT_VERSION = "1.4.0"


@dataclass
class NarrativeContract:
    """
    Contract specification for narrative generation.
    
    Defines the expected structure and content of LLM responses.
    """
    
    # Version tracking
    contract_version: str = NARRATIVE_CONTRACT_VERSION
    prompt_version: str = PROMPT_VERSION
    
    # Required sections
    required_sections: List[str] = field(default_factory=lambda: [
        "descriere_generala",
        "profil_profesional",
        "hobby_sport",
        "hobby_arta_cultura",
        "hobby_calatorii",
        "hobby_culinar",
        "career_goals_and_ambitions",
        "persona_summary",
    ])
    
    # Section markers (must be present in response)
    required_markers: List[str] = field(default_factory=lambda: [
        "[DESCRIERE GENERALA]",
        "[PROFIL PROFESIONAL]",
        "[HOBBY SPORT]",
        "[HOBBY ARTA CULTURA]",
        "[HOBBY CALATORII]",
        "[HOBBY CULINAR]",
        "[OBIECTIVE ȘI ASPIRAȚII]",
        "[REZUMAT]",
    ])
    
    # Validation rules
    min_section_length: int = 20
    max_section_length: int = 2000
    require_diacritics: bool = True
    require_name_mention: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "prompt_version": self.prompt_version,
            "required_sections": self.required_sections,
            "required_markers": self.required_markers,
            "min_section_length": self.min_section_length,
            "max_section_length": self.max_section_length,
            "require_diacritics": self.require_diacritics,
            "require_name_mention": self.require_name_mention,
        }


@dataclass
class ContractValidationResult:
    """Result of narrative contract validation."""
    
    is_valid: bool
    contract_version: str
    
    # Section-level results
    missing_sections: List[str] = field(default_factory=list)
    missing_markers: List[str] = field(default_factory=list)
    too_short_sections: List[str] = field(default_factory=list)
    too_long_sections: List[str] = field(default_factory=list)
    
    # Content validation
    missing_name: bool = False
    missing_diacritics: bool = False
    pronoun_mismatch: bool = False
    
    # Overall
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "contract_version": self.contract_version,
            "missing_sections": self.missing_sections,
            "missing_markers": self.missing_markers,
            "too_short_sections": self.too_short_sections,
            "too_long_sections": self.too_long_sections,
            "missing_name": self.missing_name,
            "missing_diacritics": self.missing_diacritics,
            "pronoun_mismatch": self.pronoun_mismatch,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


class NarrativeContractValidator:
    """
    Validates LLM responses against the narrative contract.
    
    Usage:
        validator = NarrativeContractValidator()
        result = validator.validate(response_text, persona)
        if not result.is_valid:
            raise NarrativeContractViolation(result.errors)
    """
    
    def __init__(self, contract: Optional[NarrativeContract] = None):
        self.contract = contract or NarrativeContract()
    
    def validate(self, response: str, persona: Persona) -> ContractValidationResult:
        """
        Validate a narrative response against the contract.
        
        Args:
            response: Raw LLM response text
            persona: Persona this narrative is for
            
        Returns:
            ContractValidationResult with detailed findings
        """
        result = ContractValidationResult(
            is_valid=True,
            contract_version=self.contract.contract_version,
        )
        
        # Parse response (JSON-only preferred if present)
        is_json = False
        try:
            sections = parse_json_narrative_response_strict(
                response,
                self.contract.required_sections,
                require_all=False,
            )
            is_json = True
        except Exception:
            sections = parse_narrative_response(response)
        
        # Check required sections
        for section in self.contract.required_sections:
            if not sections.get(section) or not sections[section].strip():
                result.missing_sections.append(section)
                result.errors.append(f"Missing section: {section}")
        
        # Check required markers (skip for JSON-only responses)
        if not is_json:
            for marker in self.contract.required_markers:
                if marker not in response:
                    result.missing_markers.append(marker)
                    result.errors.append(f"Missing marker: {marker}")
        
        # Check section lengths
        for section_name, content in sections.items():
            if not content:
                continue
            
            length = len(content)
            if length < self.contract.min_section_length:
                result.too_short_sections.append(section_name)
                result.warnings.append(
                    f"Section '{section_name}' is too short ({length} chars, min {self.contract.min_section_length})"
                )
            elif length > self.contract.max_section_length:
                result.too_long_sections.append(section_name)
                result.warnings.append(
                    f"Section '{section_name}' is too long ({length} chars, max {self.contract.max_section_length})"
                )
        
        # Check for Romanian diacritics
        if self.contract.require_diacritics:
            combined_text = " ".join(sections.values())
            diacritics = re.findall(r'[ăâîșțĂÂÎȘȚ]', combined_text)
            if len(diacritics) < 3:
                result.missing_diacritics = True
                result.warnings.append("Too few Romanian diacritics found")
        
        # Check name mention
        if self.contract.require_name_mention:
            combined_text = " ".join(sections.values()).lower()
            first_name = persona.name.split()[0].lower()
            if first_name not in combined_text:
                result.missing_name = True
                result.warnings.append(f"First name '{first_name}' not mentioned in narrative")
        
        # Validate against persona data
        validation_issues = validate_narrative_against_persona(sections, persona)
        for issue in validation_issues:
            if "pronoun" in issue.lower():
                result.pronoun_mismatch = True
                result.errors.append(issue)
            else:
                result.warnings.append(issue)
        
        # Determine overall validity
        # Missing sections/markers are hard errors
        # Other issues are warnings
        if result.missing_sections or result.missing_markers or result.pronoun_mismatch:
            result.is_valid = False
        
        return result


class GoldenNarrativeFixture:
    """
    Golden fixture for narrative regression testing.
    
    Captures a known-good persona + expected narrative output.
    Used to detect format drift in CI.
    """
    
    def __init__(self, persona: Persona, expected_response: str, 
                 description: str = "", fixture_version: str = "1.0"):
        self.persona = persona
        self.expected_response = expected_response
        self.description = description
        self.fixture_version = fixture_version
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixture_version": self.fixture_version,
            "description": self.description,
            "persona": self.persona.model_dump(),
            "expected_response": self.expected_response,
        }
    
    def save(self, filepath: str) -> None:
        """Save fixture to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> "GoldenNarrativeFixture":
        """Load fixture from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            persona=Persona(**data["persona"]),
            expected_response=data["expected_response"],
            description=data.get("description", ""),
            fixture_version=data.get("fixture_version", "1.0"),
        )


class NarrativeRegressionTest:
    """
    Regression test for narrative generation.
    
    Runs a fixed persona through the generator and validates
    the response structure matches expectations.
    """
    
    def __init__(self, fixture: GoldenNarrativeFixture,
                 validator: Optional[NarrativeContractValidator] = None):
        self.fixture = fixture
        self.validator = validator or NarrativeContractValidator()
    
    def run(self, generate_fn: Callable[[Persona, str], str]) -> ContractValidationResult:
        """
        Run regression test.
        
        Args:
            generate_fn: Function that takes (persona, prompt) and returns response
            
        Returns:
            Validation result
        """
        from .prompts import generate_full_prompt
        
        # Generate prompt
        prompt = generate_full_prompt(self.fixture.persona)
        
        # Generate response
        response = generate_fn(self.fixture.persona, prompt)
        
        # Validate
        return self.validator.validate(response, self.fixture.persona)


# Pre-defined golden fixtures for CI
def get_ci_regression_fixture() -> GoldenNarrativeFixture:
    """
    Get the standard regression test fixture for CI.
    
    This is a fixed persona that should always generate a valid narrative.
    If this fails, the narrative contract has drifted.
    """
    return GoldenNarrativeFixture(
        persona=Persona(
            name="Maria Popescu",
            sex="Feminin",
            age=34,
            age_group="35-44",
            ethnicity="Moldovean",
            mother_tongue="Română",
            religion="Ortodox",
            marital_status="Căsătorit",
            education_level="Superior (Licență/Master)",
            field_of_study="Economie",
            occupation="Economist",
            city="Chișinău",
            district="Mun. Chișinău",
            region="Chisinau",
            residence_type="Urban",
            country="Moldova",
        ),
        expected_response="""[DESCRIERE GENERALA]
Maria este o femeie de 34 de ani din Chișinău. Este o persoană organizată și ambițioasă, cu o atitudine pozitivă față de viață.

[PROFIL PROFESIONAL]
Lucrează ca economist. Analizează date financiare și pregătește rapoarte pentru conducere.

[HOBBY SPORT]
Îi place să facă drumeții în weekend și să practice yoga.

[HOBBY ARTA CULTURA]
Apreciază muzica clasică și citește literatură contemporană.

[HOBBY CALATORII]
Preferă călătorii culturale în Europa de Est.

[HOBBY CULINAR]
Gătește preparate tradiționale moldovenești și experimentează cu bucătăria internațională.

[OBIECTIVE ȘI ASPIRAȚII]
Își propune să avanseze profesional și să își dezvolte competențele în analiză financiară, urmărind cursuri de specializare.

[REZUMAT]
Maria Popescu, economistă de 34 de ani din Chișinău, îmbină rigoarea profesională cu interesul pentru cultură și echilibru personal.""",
        description="Standard CI regression test - adult female from Chisinau",
        fixture_version="1.0",
    )


# JSON Schema for structured output (when available)
NARRATIVE_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "descriere_generala",
        "profil_profesional",
        "hobby_sport",
        "hobby_arta_cultura",
        "hobby_calatorii",
        "hobby_culinar",
        "career_goals_and_ambitions",
        "persona_summary",
    ],
    "properties": {
        "descriere_generala": {
            "type": "string",
            "minLength": 20,
            "maxLength": 2000,
            "description": "General personality description in Romanian",
        },
        "profil_profesional": {
            "type": "string",
            "minLength": 20,
            "maxLength": 2000,
            "description": "Professional life description in Romanian",
        },
        "hobby_sport": {
            "type": "string",
            "minLength": 10,
            "maxLength": 1000,
            "description": "Sports and physical activities in Romanian",
        },
        "hobby_arta_cultura": {
            "type": "string",
            "minLength": 10,
            "maxLength": 1000,
            "description": "Cultural and artistic interests in Romanian",
        },
        "hobby_calatorii": {
            "type": "string",
            "minLength": 10,
            "maxLength": 1000,
            "description": "Travel preferences in Romanian",
        },
        "hobby_culinar": {
            "type": "string",
            "minLength": 10,
            "maxLength": 1000,
            "description": "Culinary habits in Romanian",
        },
        "career_goals_and_ambitions": {
            "type": "string",
            "minLength": 10,
            "maxLength": 2000,
            "description": "Career goals and ambitions in Romanian",
        },
        "persona_summary": {
            "type": "string",
            "minLength": 10,
            "maxLength": 400,
            "description": "One-line persona summary in Romanian",
        },
    },
    "additionalProperties": False,
}
