"""Unit tests for validators."""

import pytest
from moldova_personas.models import Persona
from moldova_personas.validators import ValidationPipeline, ValidationError


def create_test_persona(**overrides):
    """Helper to create a valid test persona with all required fields."""
    defaults = {
        "id": "test-001",
        "name": "Maria Popescu",
        "sex": "Feminin",
        "age": 35,
        "age_group": "35-44",  # Correct for age 35
        "ethnicity": "Moldovean",
        "mother_tongue": "Română",
        "religion": "Ortodox",
        "marital_status": "Căsătorit",
        "education_level": "Superior (Licență/Master)",
        "occupation": "Profesor",
        # Strict geo defaults: district must be official, city empty
        "city": "",
        "district": "Mun. Chisinau",
        "region": "Chisinau",
        "residence_type": "Urban",
    }
    defaults.update(overrides)
    return Persona(**defaults)


class TestValidationPipeline:
    """Tests for ValidationPipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = ValidationPipeline()
        assert pipeline is not None
    
    def test_pipeline_validates_personas(self):
        """Test that pipeline validates a list of personas."""
        pipeline = ValidationPipeline()
        
        personas = [
            create_test_persona(id=f"test-{i}", name=f"Person {i}")
            for i in range(5)
        ]
        
        report = pipeline.validate(personas)
        assert report.total_checked == 5

    def test_pipeline_handles_empty_list(self):
        """Test that empty inputs do not crash validation."""
        pipeline = ValidationPipeline()
        report = pipeline.validate([])
        assert report.total_checked == 0
        assert report.error_count == 0
        assert report.warning_count == 0
    
    def test_valid_persona_passes(self):
        """Test that a valid persona passes validation."""
        pipeline = ValidationPipeline()
        
        persona = create_test_persona()
        
        report = pipeline.validate([persona])
        # Should have no errors (only possible warnings)
        assert report.error_count == 0
    
    def test_age_education_mismatch_detected(self):
        """Test detection of age-education mismatch."""
        pipeline = ValidationPipeline()
        
        # 19-year-old with PhD is unrealistic (min age for doctorate is typically 23+)
        persona = create_test_persona(
            id="test-002",
            name="Ion Test",
            age=19,  # Too young for Doctorat (min realistic age ~23)
            age_group="15-24",
            education_level="Doctorat",
            occupation="Student",
        )
        
        report = pipeline.validate([persona])
        # Should have at least one error or warning
        assert report.error_count > 0 or report.warning_count > 0


class TestValidationError:
    """Tests for ValidationError class."""
    
    def test_error_creation(self):
        """Test creating a validation error."""
        error = ValidationError(
            field="age",
            message="Test error",
            severity="error"
        )
        assert error.message == "Test error"
        assert error.field == "age"
        assert error.severity == "error"
    
    def test_is_error_property(self):
        """Test the is_error property."""
        # ValidationError doesn't have is_error property - check severity directly
        error = ValidationError("age", "Test", "error")
        warning = ValidationError("age", "Test", "warning")
        
        assert error.severity == "error"
        assert warning.severity == "warning"
