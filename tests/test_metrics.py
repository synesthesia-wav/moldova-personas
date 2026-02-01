"""Tests for observability metrics."""

import pytest
import time
from unittest.mock import Mock

from moldova_personas.metrics import (
    MetricsCollector,
    get_collector,
    set_global_collector,
    timed_generation,
    record_persona_generation,
    MetricType,
)


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_counter_increment(self):
        """Test counter increment."""
        collector = MetricsCollector()
        
        collector.increment("test_counter", 1.0)
        collector.increment("test_counter", 2.0)
        
        assert collector._counters["test_counter"] == 3.0
    
    def test_gauge_set(self):
        """Test gauge value setting."""
        collector = MetricsCollector()
        
        collector.gauge("test_gauge", 42.0)
        assert collector._gauges["test_gauge"] == 42.0
        
        collector.gauge("test_gauge", 50.0)
        assert collector._gauges["test_gauge"] == 50.0
    
    def test_histogram_record(self):
        """Test histogram recording."""
        collector = MetricsCollector()
        
        collector.histogram("test_hist", 1.0)
        collector.histogram("test_hist", 2.0)
        collector.histogram("test_hist", 3.0)
        
        assert len(collector._histograms["test_hist"]) == 3
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        collector = MetricsCollector()
        
        with collector.timer("test_timer"):
            time.sleep(0.01)
        
        assert "test_timer" in collector._timers
        assert len(collector._timers["test_timer"]) == 1
        assert collector._timers["test_timer"][0] > 0
    
    def test_record_generation(self):
        """Test recording generation metrics."""
        collector = MetricsCollector()
        
        collector.record_generation(
            n=1000,
            duration=5.0,
            decision="PASS",
            profile="HIGH_STAKES",
            gate_codes=["PASS"],
            ess_ratio=0.85,
            use_ipf=True,
        )
        
        assert collector._counters["personas_generated"] == 1000
        assert "generation_duration_seconds" in collector._timers
        assert collector._gauges["ess_ratio"] == 0.85
    
    def test_get_summary(self):
        """Test summary statistics."""
        collector = MetricsCollector()
        
        collector.histogram("test", 1.0)
        collector.histogram("test", 2.0)
        collector.histogram("test", 3.0)
        collector.histogram("test", 4.0)
        collector.histogram("test", 5.0)
        
        summary = collector.get_summary()
        
        assert "histograms" in summary
        assert "test" in summary["histograms"]
        assert summary["histograms"]["test"]["count"] == 5
        assert summary["histograms"]["test"]["mean"] == 3.0
    
    def test_custom_backend(self):
        """Test custom backend callback."""
        backend_calls = []
        
        def mock_backend(metric):
            backend_calls.append(metric)
        
        collector = MetricsCollector()
        collector.set_backend(mock_backend)
        
        collector.increment("test", 1.0, {"tag": "value"})
        
        assert len(backend_calls) == 1
        assert backend_calls[0].name == "test"
        assert backend_calls[0].value == 1.0
        assert backend_calls[0].tags == {"tag": "value"}


class TestGlobalCollector:
    """Tests for global collector functions."""
    
    def test_get_collector_singleton(self):
        """Test that get_collector returns singleton."""
        c1 = get_collector()
        c2 = get_collector()
        
        assert c1 is c2
    
    def test_set_global_collector(self):
        """Test setting global collector."""
        new_collector = MetricsCollector()
        
        set_global_collector(new_collector)
        
        assert get_collector() is new_collector
    
    def test_timed_generation_decorator(self):
        """Test timed_generation context manager."""
        collector = MetricsCollector()
        set_global_collector(collector)
        
        with timed_generation("my_generation", profile="test"):
            time.sleep(0.01)
        
        assert "my_generation" in collector._timers


class TestRecordPersonaGeneration:
    """Tests for record_persona_generation function."""
    
    def test_record_function(self):
        """Test the convenience function."""
        collector = MetricsCollector()
        set_global_collector(collector)
        
        record_persona_generation(
            n=500,
            duration=2.5,
            decision="PASS",
            profile="ANALYSIS_ONLY",
            gate_codes=["PASS", "ESS_OK"],
        )
        
        assert collector._counters["personas_generated"] == 500
        assert "generation_duration_seconds" in collector._timers
