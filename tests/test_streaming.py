"""Tests for streaming generation and export."""

import pytest
from pathlib import Path
import tempfile

from moldova_personas import (
    StreamingGenerator,
    StreamingParquetExporter,
    StreamingJSONLExporter,
    generate_and_export_streaming,
)
from moldova_personas.models import Persona


class TestStreamingGenerator:
    """Tests for streaming generation."""
    
    def test_streaming_generator_creates_personas(self):
        """Test that streaming generator creates valid personas."""
        streamer = StreamingGenerator(seed=42)
        
        personas = list(streamer.generate_single_stream(n=100))
        
        assert len(personas) == 100
        assert all(isinstance(p, Persona) for p in personas)
    
    def test_streaming_batches(self):
        """Test batch generation."""
        streamer = StreamingGenerator(seed=42)
        
        batches = list(streamer.generate_batches(n=1000, batch_size=250))
        
        assert len(batches) == 4
        assert sum(len(b) for b in batches) == 1000
        assert all(isinstance(p, Persona) for batch in batches for p in batch)
    
    def test_streaming_memory_efficiency(self):
        """Test that streaming doesn't accumulate all personas."""
        streamer = StreamingGenerator(seed=42)
        
        count = 0
        for batch in streamer.generate_batches(n=1000, batch_size=100):
            count += len(batch)
            # Each batch should be processed and freed
            del batch
        
        assert count == 1000


class TestStreamingExporters:
    """Tests for streaming exporters."""
    
    def test_parquet_streaming_export(self):
        """Test streaming Parquet export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.parquet"
            
            streamer = StreamingGenerator(seed=42)
            
            with StreamingParquetExporter(str(filepath)) as exporter:
                for batch in streamer.generate_batches(n=1000, batch_size=250):
                    exporter.write_batch(batch)
            
            assert filepath.exists()
            assert filepath.stat().st_size > 0
    
    def test_jsonl_streaming_export(self):
        """Test streaming JSONL export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            
            streamer = StreamingGenerator(seed=42)
            
            with StreamingJSONLExporter(str(filepath)) as exporter:
                for batch in streamer.generate_batches(n=100, batch_size=25):
                    exporter.write_batch(batch)
            
            assert filepath.exists()
            
            # Check number of lines
            lines = filepath.read_text().strip().split('\n')
            assert len(lines) == 100
    
    def test_generate_and_export_streaming(self):
        """Test combined generation and export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "output.parquet"
            
            progress_calls = []
            
            def progress(complete, total):
                progress_calls.append((complete, total))
            
            result = generate_and_export_streaming(
                n=500,
                output_path=str(filepath),
                batch_size=100,
                seed=42,
                format="parquet",
                progress_callback=progress,
            )
            
            assert filepath.exists()
            assert result["total_generated"] == 500
            assert result["duration_seconds"] > 0
            assert len(progress_calls) > 0


class TestStreamingAgeEducation:
    """Test age-education consistency in streaming mode."""
    
    def test_streaming_no_age_education_violations(self):
        """Test that streaming generation respects age-education constraints."""
        streamer = StreamingGenerator(seed=42)
        
        violations = 0
        total = 0
        
        for batch in streamer.generate_batches(n=1000, batch_size=100):
            for p in batch:
                total += 1
                if p.education_level == "Superior (Licență/Master)" and p.age < 19:
                    violations += 1
                elif p.education_level == "Doctorat" and p.age < 27:
                    violations += 1
        
        assert violations == 0, f"Found {violations} age-education violations in {total} personas"
