"""Tests for async narrative generator."""

import json
import pytest
import time
from unittest.mock import Mock, patch

from moldova_personas.async_narrative_generator import (
    AsyncNarrativeGenerator,
    GenerationResult,
    enrich_personas_parallel
)
from moldova_personas import PersonaGenerator


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.call_count = 0
    
    def generate(self, prompt: str, config=None):
        self.call_count += 1
        time.sleep(self.delay)
        return f"Descriere generala: Test persona\\nProfil profesional: Test job\\nHobby sport: Fotbal\\nHobby arta: Muzica\\nHobby calatorii: Europa\\nHobby culinar: Traditional"


class TestAsyncNarrativeGenerator:
    """Tests for AsyncNarrativeGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = AsyncNarrativeGenerator(
            provider="mock",
            max_workers=5,
            rate_limit_per_second=2.0
        )
        
        assert gen.max_workers == 5
        assert gen.rate_limit == 2.0
        assert gen.stats['total'] == 0
    
    def test_parallel_faster_than_serial(self):
        """Test that parallel processing is actually faster."""
        # Create mock client with 0.1s delay
        mock_client = MockLLMClient(delay=0.1)
        
        # Generate test personas
        persona_gen = PersonaGenerator(seed=42)
        personas = persona_gen.generate(10, show_progress=False)
        
        # Serial processing
        gen_serial = AsyncNarrativeGenerator(
            llm_client=mock_client,
            max_workers=1
        )
        
        mock_client.call_count = 0
        start = time.time()
        gen_serial.generate_batch(personas, show_progress=False)
        serial_time = time.time() - start
        
        # Parallel processing
        gen_parallel = AsyncNarrativeGenerator(
            llm_client=MockLLMClient(delay=0.1),
            max_workers=5
        )
        
        start = time.time()
        gen_parallel.generate_batch(personas, show_progress=False)
        parallel_time = time.time() - start
        
        # Parallel should be significantly faster
        # Expected: serial ~1.0s, parallel ~0.2s (5 workers)
        speedup = serial_time / parallel_time
        assert speedup > 2.0, f"Parallel speedup only {speedup:.1f}x, expected >2x"
    
    def test_generation_result(self):
        """Test GenerationResult dataclass."""
        from moldova_personas import PersonaGenerator
        
        gen = PersonaGenerator(seed=42)
        persona = gen.generate_single()
        
        result = GenerationResult(
            persona=persona,
            success=True,
            latency_ms=150.5
        )
        
        assert result.success is True
        assert result.latency_ms == 150.5
        assert result.error is None
    
    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        mock_client = MockLLMClient(delay=0.01)
        gen = AsyncNarrativeGenerator(llm_client=mock_client, max_workers=2)
        
        persona_gen = PersonaGenerator(seed=42)
        personas = persona_gen.generate(5, show_progress=False)
        
        gen.generate_batch(personas, show_progress=False)
        stats = gen.get_stats()
        
        assert stats['total'] == 5
        assert stats['success'] == 5
        assert stats['failed'] == 0
        assert 'avg_latency_ms' in stats
        assert stats['success_rate'] == 1.0
    
    def test_error_handling(self):
        """Test that errors are handled gracefully."""
        class FailingClient:
            def generate(self, prompt, config=None):
                raise Exception("API Error")
        
        gen = AsyncNarrativeGenerator(llm_client=FailingClient(), max_workers=2)
        
        persona_gen = PersonaGenerator(seed=42)
        personas = persona_gen.generate(3, show_progress=False)
        
        # Should not raise, should return originals
        results = gen.generate_batch(personas, show_progress=False)
        
        assert len(results) == 3
        stats = gen.get_stats()
        assert stats['failed'] == 3
        assert stats['success'] == 0
    
    def test_rate_limiting(self):
        """Test that rate limiting works."""
        mock_client = MockLLMClient(delay=0.01)
        gen = AsyncNarrativeGenerator(
            llm_client=mock_client,
            max_workers=10,
            rate_limit_per_second=5.0  # 5 requests/sec max
        )
        
        persona_gen = PersonaGenerator(seed=42)
        personas = persona_gen.generate(5, show_progress=False)
        
        start = time.time()
        gen.generate_batch(personas, show_progress=False)
        elapsed = time.time() - start
        
        # With 5 requests at 5/sec, minimum time is ~0.8s
        # Allow some tolerance
        assert elapsed >= 0.5, f"Rate limiting not working: {elapsed:.2f}s for 5 requests"
    
    def test_convenience_function(self):
        """Test enrich_personas_parallel convenience function."""
        persona_gen = PersonaGenerator(seed=42)
        personas = persona_gen.generate(3, show_progress=False)
        
        # Should work with mock provider
        results = enrich_personas_parallel(
            personas,
            provider="mock",
            max_workers=2,
            show_progress=False
        )
        
        assert len(results) == 3


class TestParallelPerformance:
    """Performance tests for parallel processing."""
    
    @pytest.mark.slow
    def test_speedup_with_workers(self):
        """Test speedup scales with worker count."""
        delays = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        json_response = json.dumps({
            "descriere_generala": "x" * 25,
            "profil_profesional": "y" * 25,
            "hobby_sport": "z" * 15,
            "hobby_arta_cultura": "w" * 15,
            "hobby_calatorii": "v" * 15,
            "hobby_culinar": "u" * 15,
            "career_goals_and_ambitions": "t" * 15,
            "persona_summary": "s" * 15,
        })
        
        class VariableDelayClient:
            def __init__(self, delays):
                self.delays = delays
                self.index = 0
            
            def generate(self, prompt, config=None):
                delay = self.delays[self.index % len(self.delays)]
                self.index += 1
                time.sleep(delay)
                return json_response
        
        persona_gen = PersonaGenerator(seed=42)
        personas = persona_gen.generate(8, show_progress=False)
        
        # Test with different worker counts
        for workers in [1, 2, 4]:
            client = VariableDelayClient(delays)
            gen = AsyncNarrativeGenerator(llm_client=client, max_workers=workers)
            
            start = time.time()
            gen.generate_batch(personas, show_progress=False)
            elapsed = time.time() - start
            
            # Expected time: (total_delay / workers) + overhead
            expected_approx = (sum(delays) / workers) + 0.1
            
            assert elapsed < expected_approx * 1.5, (
                f"With {workers} workers, took {elapsed:.2f}s, "
                f"expected ~{expected_approx:.2f}s"
            )
