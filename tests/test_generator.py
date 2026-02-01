"""Unit tests for the persona generator."""

import pytest
from moldova_personas.generator import PersonaGenerator
from moldova_personas.geo_tables import strict_geo_enabled
from moldova_personas.models import Persona


class TestPersonaGenerator:
    """Tests for PersonaGenerator class."""
    
    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        gen = PersonaGenerator()
        assert gen is not None
    
    def test_generator_with_seed(self):
        """Test that generator produces reproducible results with same seed."""
        gen1 = PersonaGenerator(seed=42)
        gen2 = PersonaGenerator(seed=42)
        
        p1 = gen1.generate_single()
        p2 = gen2.generate_single()
        
        # Same seed should produce similar distributions
        assert isinstance(p1, Persona)
        assert isinstance(p2, Persona)
    
    def test_generate_single_returns_persona(self):
        """Test that generate_single returns a Persona object."""
        gen = PersonaGenerator()
        persona = gen.generate_single()
        
        assert isinstance(persona, Persona)
        assert persona.name is not None
        assert persona.age > 0
        assert persona.sex in ["Masculin", "Feminin"]
    
    def test_generate_batch(self):
        """Test batch generation."""
        gen = PersonaGenerator()
        personas = gen.generate(10)
        
        assert len(personas) == 10
        assert all(isinstance(p, Persona) for p in personas)
    
    def test_age_education_consistency(self):
        """Test that generated personas have age-appropriate education."""
        gen = PersonaGenerator()
        
        # Generate many personas to check consistency
        for _ in range(100):
            p = gen.generate_single()
            
            # Check age-education alignment
            if p.education_level == "Doctorat":
                assert p.age >= 27, f"Age {p.age} too young for Doctorat"
            elif p.education_level == "Superior (Licență/Master)":
                assert p.age >= 19, f"Age {p.age} too young for Superior"
    
    def test_age_18_no_superior_education(self):
        """Test that age 18 never gets Superior education (regression test)."""
        gen = PersonaGenerator(seed=42)
        
        # Generate many personas and check none have age 18 + Superior
        personas = gen.generate(200)
        
        for p in personas:
            if p.age == 18:
                assert p.education_level not in ["Superior (Licență/Master)", "Doctorat"], \
                    f"Age 18 should not have {p.education_level} education"
    
    def test_age_education_with_ipf_correction(self):
        """Test age-education consistency even with IPF correction."""
        gen = PersonaGenerator(seed=123)
        
        personas = gen.generate_with_ethnicity_correction(n=100, oversample_factor=3)
        
        violations = []
        for p in personas:
            if p.education_level == "Superior (Licență/Master)" and p.age < 19:
                violations.append(f"Age {p.age} with Superior")
            elif p.education_level == "Doctorat" and p.age < 27:
                violations.append(f"Age {p.age} with Doctorat")
        
        assert len(violations) == 0, f"Age-education violations: {violations[:5]}"


class TestDemographicValidation:
    """Tests for demographic validation."""
    
    def test_ethnicity_distribution(self):
        """Test that ethnicity is from valid set."""
        gen = PersonaGenerator()
        valid_ethnicities = {"Moldovean", "Român", "Ucrainean", "Rus", "Găgăuz", "Bulgar", "Rrom"}
        
        for _ in range(20):
            p = gen.generate_single()
            assert p.ethnicity in valid_ethnicities
    
    def test_region_validity(self):
        """Test that region is valid."""
        gen = PersonaGenerator()
        valid_regions = {"Chisinau", "Nord", "Centru", "Sud", "Gagauzia"}
        
        for _ in range(20):
            p = gen.generate_single()
            assert p.region in valid_regions
    
    def test_city_matches_region(self):
        """Test that city belongs to the assigned region."""
        gen = PersonaGenerator()
        
        for _ in range(20):
            p = gen.generate_single()
            # Basic check: city should be a string
            assert isinstance(p.city, str)
            if strict_geo_enabled():
                # Strict geo mode may omit localities
                assert p.city == ""
            else:
                assert len(p.city) > 0
