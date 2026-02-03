"""Tests for prompt generation and parsing."""

import pytest

from moldova_personas.prompts import (
    generate_full_prompt,
    parse_narrative_response,
    validate_narrative_against_persona,
    get_prompt_version,
    _get_sex_pronouns,
    PROMPT_VERSION,
)
from moldova_personas import PersonaGenerator


class TestPromptVersion:
    """Tests for prompt versioning."""
    
    def test_prompt_version_exists(self):
        """Test that prompt version is defined."""
        version = get_prompt_version()
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0
    
    def test_prompt_version_format(self):
        """Test that version follows semver format."""
        version = get_prompt_version()
        parts = version.split('.')
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestPronouns:
    """Tests for pronoun generation."""
    
    def test_feminine_pronouns(self):
        """Test correct feminine pronouns."""
        pronouns = _get_sex_pronouns("Feminin")
        assert pronouns["subject"] == "ea"
        assert pronouns["possessive"] == "ei"
    
    def test_masculine_pronouns(self):
        """Test correct masculine pronouns."""
        pronouns = _get_sex_pronouns("Masculin")
        assert pronouns["subject"] == "el"
        assert pronouns["possessive"] == "lui"


class TestPromptGeneration:
    """Tests for prompt generation."""
    
    def test_prompt_contains_version(self):
        """Test that prompt contains version."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        prompt = generate_full_prompt(persona)
        assert PROMPT_VERSION in prompt
    
    def test_prompt_contains_persona_data(self):
        """Test that prompt contains persona details."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        prompt = generate_full_prompt(persona)
        assert persona.name in prompt
        assert str(persona.age) in prompt
        assert persona.region in prompt
        assert persona.ethnicity in prompt
    
    def test_prompt_contains_pronoun_instruction(self):
        """Test that prompt explicitly mentions pronouns."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        prompt = generate_full_prompt(persona)
        assert "pronume" in prompt.lower()
    
    def test_prompt_requires_region_reference(self):
        """Test that prompt requires region reference."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        prompt = generate_full_prompt(persona)
        # Should have explicit instruction about region
        assert "obligatoriu" in prompt.lower() or "trebuie" in prompt.lower()
    
    def test_prompt_requires_ethnicity_marker(self):
        """Test that prompt requires ethnicity marker."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        prompt = generate_full_prompt(persona)
        # Should mention ethnicity
        assert persona.ethnicity in prompt
        assert "marker" in prompt.lower() or "cultural" in prompt.lower()
    
    def test_adult_prompt_structure(self):
        """Test adult prompt has correct sections."""
        gen = PersonaGenerator(seed=42)
        # Filter for adult
        adult = None
        for _ in range(20):
            p = gen.generate_single()
            if 18 <= p.age < 65:
                adult = p
                break
        
        if adult:
            prompt = generate_full_prompt(adult)
            assert "FORMAT DE RĂSPUNS (JSON ONLY)" in prompt or "FORMAT DE RASPUNS (JSON ONLY)" in prompt
            assert '"descriere_generala"' in prompt
            assert '"profil_profesional"' in prompt
            assert '"hobby_sport"' in prompt
    
    def test_elderly_prompt_structure(self):
        """Test elderly prompt mentions retirement."""
        gen = PersonaGenerator(seed=42)
        # Filter for elderly
        elderly = None
        for _ in range(50):
            p = gen.generate_single()
            if p.age >= 65:
                elderly = p
                break
        
        if elderly:
            prompt = generate_full_prompt(elderly)
            assert "pensie" in prompt.lower() or "pensionar" in prompt.lower()


class TestResponseParsing:
    """Tests for narrative response parsing."""
    
    def test_parse_markers(self):
        """Test parsing with explicit markers."""
        response = """
[DESCRIERE GENERALA]
Maria este o femeie de 45 de ani.

[PROFIL PROFESIONAL]
Lucrează ca profesoară.

[HOBBY SPORT]
Îi place să alerge.

[HOBBY ARTA CULTURA]
Îi place muzica clasică.

[HOBBY CALATORII]
A călătorit în Italia.

[HOBBY CULINAR]
Gătește tradițional.
"""
        result = parse_narrative_response(response)
        
        assert "Maria este o femeie" in result["descriere_generala"]
        assert "profesoară" in result["profil_profesional"]
        assert "alerge" in result["hobby_sport"]
    
    def test_parse_headers(self):
        """Test parsing with header-style sections."""
        response = """
**Descriere generală**
Ion este un om simplu.

**Profil profesional**
Lucrează în construcții.

**Hobby-uri sportive**
Joacă fotbal.
"""
        result = parse_narrative_response(response)
        
        assert "Ion este" in result["descriere_generala"]
        assert "construcții" in result["profil_profesional"]
    
    def test_parse_fallback(self):
        """Test fallback when no markers found."""
        response = "Acesta este un text fără marcaje speciale."
        result = parse_narrative_response(response)
        
        assert result["descriere_generala"] == response.strip()
    
    def test_parse_empty_response(self):
        """Test parsing empty response."""
        result = parse_narrative_response("")
        assert result["descriere_generala"] == ""


class TestNarrativeValidation:
    """Tests for narrative validation against persona."""
    
    def test_validation_detects_missing_age(self):
        """Test validation catches missing age."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        # Narrative without age
        narrative = {
            "descriere_generala": "O persoană care locuiește în Chișinău.",
            "profil_profesional": "",
            "hobby_sport": "",
            "hobby_arta_cultura": "",
            "hobby_calatorii": "",
            "hobby_culinar": "",
        }
        
        issues = validate_narrative_against_persona(narrative, persona)
        assert any("age" in i.lower() or "ani" in i.lower() for i in issues)
    
    def test_validation_detects_wrong_pronoun_female(self):
        """Test validation catches wrong pronoun for female."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        persona.sex = "Feminin"
        persona.name = "Maria Popescu"
        persona.age = 35
        persona.region = "Nord"
        persona.occupation = "Profesor"
        
        # Narrative with male pronoun
        narrative = {
            "descriere_generala": "el este Maria de 35 de ani din Nord. Lucrează ca Profesor.",
            "profil_profesional": "",
            "hobby_sport": "",
            "hobby_arta_cultura": "",
            "hobby_calatorii": "",
            "hobby_culinar": "",
        }
        
        issues = validate_narrative_against_persona(narrative, persona)
        # Should detect male pronoun
        assert any("male" in i.lower() for i in issues)
    
    def test_validation_detects_wrong_pronoun_male(self):
        """Test validation catches wrong pronoun for male."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        persona.sex = "Masculin"
        persona.name = "Ion Popescu"
        persona.age = 40
        persona.region = "Chisinau"
        persona.occupation = "Inginer"
        
        # Narrative with female pronoun
        narrative = {
            "descriere_generala": "ea este Ion de 40 de ani din Chisinau. Lucrează ca Inginer.",
            "profil_profesional": "",
            "hobby_sport": "",
            "hobby_arta_cultura": "",
            "hobby_calatorii": "",
            "hobby_culinar": "",
        }
        
        issues = validate_narrative_against_persona(narrative, persona)
        # Should detect female pronoun
        assert any("female" in i.lower() for i in issues)
    
    def test_validation_passes_correct_narrative(self):
        """Test validation passes correct narrative."""
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        # Correct narrative
        pronoun = "ea" if persona.sex == "Feminin" else "el"
        narrative = {
            "descriere_generala": f"{pronoun} este {persona.name} de {persona.age} ani din {persona.region}.",
            "profil_profesional": f"Lucrează ca {persona.occupation}.",
            "hobby_sport": "",
            "hobby_arta_cultura": "",
            "hobby_calatorii": "",
            "hobby_culinar": "",
        }
        
        issues = validate_narrative_against_persona(narrative, persona)
        # Should have minimal or no issues
        assert len(issues) <= 1
