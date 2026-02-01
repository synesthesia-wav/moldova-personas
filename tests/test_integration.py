"""Integration tests for the persona generation pipeline.

These tests verify end-to-end functionality including:
- Full generation pipeline with validation
- Export functionality
- LLM client factory
"""

import tempfile
import json
from pathlib import Path

import pytest

from moldova_personas import PersonaGenerator, ValidationPipeline, ParquetExporter
from moldova_personas.llm_client import create_llm_client, MockLLMClient
from moldova_personas.exporters import export_all_formats, JSONExporter, CSVExporter
from moldova_personas.models import Persona, AgeConstraints


class TestFullPipeline:
    """End-to-end pipeline integration tests."""
    
    def test_generate_and_validate(self):
        """Test full generation and validation pipeline."""
        gen = PersonaGenerator(seed=123)
        personas = gen.generate(50)
        
        # Validate
        validator = ValidationPipeline()
        report = validator.validate(personas)
        
        assert report.is_valid, f"Validation failed: {report.errors}"
        assert report.total_checked == 50
    
    def test_age_constraints_respected(self):
        """Verify all generated personas respect AgeConstraints."""
        gen = PersonaGenerator(seed=456)
        personas = gen.generate(100)
        
        for p in personas:
            # All ages should be >= MIN_PERSONA_AGE
            assert p.age >= AgeConstraints.MIN_PERSONA_AGE, \
                f"Age {p.age} below minimum {AgeConstraints.MIN_PERSONA_AGE}"
            
            # Education-age consistency
            if p.education_level == "Superior (Licență/Master)":
                assert p.age >= AgeConstraints.MIN_HIGHER_EDUCATION, \
                    f"Age {p.age} too young for {p.education_level}"
            elif p.education_level == "Doctorat":
                assert p.age >= AgeConstraints.MIN_DOCTORATE, \
                    f"Age {p.age} too young for {p.education_level}"
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical batch results."""
        # Generate batch with seed
        gen1 = PersonaGenerator(seed=999)
        batch1 = gen1.generate(10, show_progress=False)
        
        # Generate another batch with same seed
        gen2 = PersonaGenerator(seed=999)
        batch2 = gen2.generate(10, show_progress=False)
        
        # Batches should be identical
        for p1, p2 in zip(batch1, batch2):
            assert p1.name == p2.name
            assert p1.age == p2.age
            assert p1.sex == p2.sex
            assert p1.region == p2.region
            assert p1.ethnicity == p2.ethnicity
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = PersonaGenerator(seed=111)
        gen2 = PersonaGenerator(seed=222)
        
        batch1 = gen1.generate(10, show_progress=False)
        batch2 = gen2.generate(10, show_progress=False)
        
        # At least some personas should be different
        all_same = all(
            p1.name == p2.name and p1.age == p2.age
            for p1, p2 in zip(batch1, batch2)
        )
        assert not all_same, "Different seeds produced identical batches"


class TestExportFunctionality:
    """Tests for export functionality."""
    
    def test_parquet_export(self):
        """Test Parquet export creates valid file."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.parquet"
            exporter = ParquetExporter()
            exporter.export(personas, str(filepath))
            
            assert filepath.exists()
            assert filepath.stat().st_size > 0
            
            # Verify can be read back
            import pandas as pd
            df = pd.read_parquet(filepath)
            assert len(df) == 10
            assert 'name' in df.columns
            assert 'age' in df.columns
    
    def test_json_export(self):
        """Test JSON export creates valid file."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            exporter = JSONExporter()
            exporter.export_json(personas, str(filepath))
            
            assert filepath.exists()
            
            # Verify JSON structure
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert len(data) == 5
            assert all('uuid' in p for p in data)
            assert all('name' in p for p in data)
    
    def test_csv_export(self):
        """Test CSV export creates valid file."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            exporter = CSVExporter()
            exporter.export(personas, str(filepath))
            
            assert filepath.exists()
            
            # Verify CSV structure
            import pandas as pd
            df = pd.read_csv(filepath)
            assert len(df) == 5
            assert 'name' in df.columns
    
    def test_export_all_formats(self):
        """Test export_all_formats creates all expected files."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = export_all_formats(personas, tmpdir, basename="test")
            
            assert 'parquet' in results
            assert 'json' in results
            assert 'jsonl' in results
            assert 'csv' in results
            assert 'stats_json' in results
            assert 'stats_markdown' in results
            
            # Verify files exist
            for path in results.values():
                assert Path(path).exists()


class TestLLMClientFactory:
    """Tests for LLM client factory."""
    
    def test_create_mock_client(self):
        """Test factory creates mock client."""
        client = create_llm_client("mock")
        assert isinstance(client, MockLLMClient)
    
    def test_mock_client_generation(self):
        """Test mock client returns empty string (no fabricated content)."""
        client = create_llm_client("mock", delay=0)
        response = client.generate("Test prompt")
        # Mock client now returns empty string to avoid fabricated data
        assert response == ""
    
    def test_unknown_provider_raises(self):
        """Test factory raises for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client("unknown_provider")


class TestPersonaModel:
    """Tests for Persona Pydantic model."""
    
    def test_persona_creation(self):
        """Test Persona can be created with required fields."""
        p = Persona(
            name="Maria Popescu",
            sex="Feminin",
            age=35,
            age_group="35-44",
            ethnicity="Moldovean",
            mother_tongue="Română",
            religion="Ortodox",
            marital_status="Căsătorit",
            education_level="Superior (Licență/Master)",
            occupation="Profesor",
            city="Chișinău",
            district="Chișinău",
            region="Chisinau",
            residence_type="Urban"
        )
        
        assert p.name == "Maria Popescu"
        assert p.age == 35
        assert p.uuid is not None
    
    def test_persona_serialization(self):
        """Test Persona can be serialized and deserialized."""
        p = Persona(
            name="Ion Ciobanu",
            sex="Masculin",
            age=45,
            age_group="45-54",
            ethnicity="Român",
            mother_tongue="Română",
            religion="Ortodox",
            marital_status="Căsătorit",
            education_level="Profesional/Tehnic",
            occupation="Mecanic",
            city="Bălți",
            district="Bălți",
            region="Nord",
            residence_type="Urban"
        )
        
        # Serialize
        data = p.model_dump()
        
        # Deserialize
        p2 = Persona(**data)
        
        assert p2.name == p.name
        assert p2.age == p.age
        assert p2.uuid == p.uuid
    
    def test_age_constraints_class(self):
        """Test AgeConstraints constants are accessible."""
        assert AgeConstraints.MIN_PERSONA_AGE == 18
        assert AgeConstraints.MIN_HIGHER_EDUCATION == 19
        assert AgeConstraints.MIN_DOCTORATE == 27
        assert AgeConstraints.MAX_REALISTIC_AGE == 90
