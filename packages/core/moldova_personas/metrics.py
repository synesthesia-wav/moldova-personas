"""
Observability Metrics for Moldova Personas Generator

Provides hooks for Prometheus, Datadog, or custom metrics backends.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


class MetricsCollector:
    """
    Collector for generation metrics.
    
    Usage:
        collector = MetricsCollector()
        
        with collector.timer("generation_duration"):
            personas = generator.generate(1000)
        
        collector.record_generation(n=1000, decision="PASS")
        
        # Export metrics
        metrics = collector.export()
    """
    
    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._timers: Dict[str, List[float]] = {}
        self._custom_backend: Optional[Callable[[MetricValue], None]] = None
    
    def set_backend(self, backend: Callable[[MetricValue], None]):
        """Set a custom backend for real-time metric export."""
        self._custom_backend = backend
    
    def _emit(self, metric: MetricValue):
        """Emit metric to custom backend if configured."""
        if self._custom_backend:
            try:
                self._custom_backend(metric)
            except Exception as e:
                logger.warning(f"Failed to emit metric {metric.name}: {e}")
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value
        self._emit(MetricValue(name, value, MetricType.COUNTER, tags or {}))
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        self._gauges[name] = value
        self._emit(MetricValue(name, value, MetricType.GAUGE, tags or {}))
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
        self._emit(MetricValue(name, value, MetricType.HISTOGRAM, tags or {}))
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Time a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(duration)
            self._emit(MetricValue(name, duration, MetricType.TIMER, tags or {}))
    
    def record_generation(
        self,
        n: int,
        duration: float,
        decision: str,
        profile: str,
        gate_codes: List[str],
        ess_ratio: Optional[float] = None,
        use_ipf: bool = False,
    ):
        """Record a complete generation run."""
        tags = {
            "decision": decision,
            "profile": profile,
            "use_ipf": str(use_ipf),
        }
        
        self.increment("personas_generated", n, tags)
        # Record duration as both histogram (distribution) and timer (for tests/summary)
        self.histogram("generation_duration_seconds", duration, tags)
        timer_name = "generation_duration_seconds"
        if timer_name not in self._timers:
            self._timers[timer_name] = []
        self._timers[timer_name].append(duration)
        self._emit(MetricValue(timer_name, duration, MetricType.TIMER, tags))
        self.increment(f"gate_decision_total", 1, {**tags, "decision": decision})
        
        for code in gate_codes:
            self.increment("gate_code_triggered", 1, {**tags, "code": code})
        
        if ess_ratio is not None:
            self.gauge("ess_ratio", ess_ratio, tags)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        def stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"count": 0, "min": 0, "max": 0, "mean": 0, "p95": 0}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            p95_idx = int(n * 0.95)
            return {
                "count": n,
                "min": sorted_vals[0],
                "max": sorted_vals[-1],
                "mean": sum(sorted_vals) / n,
                "p95": sorted_vals[min(p95_idx, n - 1)],
            }
        
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: stats(v) for k, v in self._histograms.items()},
            "timers": {k: stats(v) for k, v in self._timers.items()},
        }
    
    def export(self) -> List[MetricValue]:
        """Export all metrics as list."""
        metrics = []
        
        for name, value in self._counters.items():
            metrics.append(MetricValue(name, value, MetricType.COUNTER))
        
        for name, value in self._gauges.items():
            metrics.append(MetricValue(name, value, MetricType.GAUGE))
        
        for name, values in self._histograms.items():
            for v in values:
                metrics.append(MetricValue(name, v, MetricType.HISTOGRAM))
        
        for name, values in self._timers.items():
            for v in values:
                metrics.append(MetricValue(name, v, MetricType.TIMER))
        
        return metrics
    
    def reset(self):
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timers.clear()


# Global collector instance
_GLOBAL_COLLECTOR: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is None:
        _GLOBAL_COLLECTOR = MetricsCollector()
    return _GLOBAL_COLLECTOR


def set_global_collector(collector: MetricsCollector):
    """Set global metrics collector."""
    global _GLOBAL_COLLECTOR
    _GLOBAL_COLLECTOR = collector


@contextmanager
def timed_generation(name: str = "generation", **tags):
    """Context manager for timing generation."""
    collector = get_collector()
    with collector.timer(name, tags):
        yield


def record_persona_generation(
    n: int,
    duration: float,
    decision: str,
    profile: str,
    gate_codes: List[str],
    **kwargs
):
    """Record generation metrics."""
    collector = get_collector()
    collector.record_generation(
        n=n,
        duration=duration,
        decision=decision,
        profile=profile,
        gate_codes=gate_codes,
        **kwargs
    )


# Example Prometheus-compatible formatter
def to_prometheus_format(metrics: List[MetricValue]) -> str:
    """Convert metrics to Prometheus exposition format."""
    lines = []
    
    for m in metrics:
        tags = ",".join(f'{k}="{v}"' for k, v in m.tags.items())
        if tags:
            lines.append(f"{m.name}{{{tags}}} {m.value}")
        else:
            lines.append(f"{m.name} {m.value}")
    
    return "\n".join(lines)
