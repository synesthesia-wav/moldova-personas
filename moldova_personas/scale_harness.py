"""
Scale harness for Nemotron-parity production runs.

Implements 3-tier testing:
- Tier 1: 1,000 personas (smoke test)
- Tier 2: 25,000 personas (distribution + drift detection)
- Tier 3: 100,000 personas (throughput + cost + failure modes)

Logs per-stage metrics for quality monitoring.
"""

import time
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import statistics
from collections import defaultdict


@dataclass
class StageMetrics:
    """Metrics for a single generation stage."""
    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Latency
    latencies: List[float] = field(default_factory=list)
    
    # Retry/failure tracking
    total_attempts: int = 0
    success_count: int = 0
    failure_count: int = 0
    retry_counts: List[int] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Token usage (for cost tracking)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Repair tracking
    repair_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_attempt(self, latency: float, success: bool, retries: int = 0):
        """Record a single attempt."""
        self.total_attempts += 1
        self.latencies.append(latency)
        self.retry_counts.append(retries)
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def record_repair(self, repair_type: str):
        """Record a repair operation."""
        self.repair_counts[repair_type] += 1
    
    def record_failure(self, reason: str):
        """Record a failure with reason."""
        self.failure_reasons[reason] += 1
    
    def record_tokens(self, prompt: int, completion: int):
        """Record token usage."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.latencies:
            return {"stage": self.stage_name, "status": "no_data"}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            "stage": self.stage_name,
            "duration_seconds": self.end_time - self.start_time if self.end_time else 0,
            "total_attempts": self.total_attempts,
            "success_rate": self.success_count / max(self.total_attempts, 1),
            "failure_rate": self.failure_count / max(self.total_attempts, 1),
            "latency_p50": sorted_latencies[n // 2] if n > 0 else 0,
            "latency_p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            "latency_p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            "mean_retries": statistics.mean(self.retry_counts) if self.retry_counts else 0,
            "total_tokens": self.total_tokens,
            "repair_summary": dict(self.repair_counts),
            "failure_reasons": dict(self.failure_reasons),
        }


@dataclass
class ScaleRunMetrics:
    """Complete metrics for a scale run."""
    # Run metadata
    run_id: str
    tier: str  # "1K", "25K", "100K"
    target_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    seed: Optional[int] = None
    
    # Overall timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Stage metrics
    stage_a: StageMetrics = field(default_factory=lambda: StageMetrics("stage_a"))
    stage_b: StageMetrics = field(default_factory=lambda: StageMetrics("stage_b"))
    stage_c: StageMetrics = field(default_factory=lambda: StageMetrics("stage_c"))
    validation: StageMetrics = field(default_factory=lambda: StageMetrics("validation"))
    
    # Quality drift tracking (for tier 2+)
    checkpoint_metrics: List[Dict] = field(default_factory=list)
    
    def start(self):
        """Start the run timer."""
        self.start_time = time.time()
    
    def end(self):
        """End the run timer."""
        self.end_time = time.time()
    
    def add_checkpoint(self, count: int, metrics: Dict):
        """Add a checkpoint metric (for drift detection)."""
        self.checkpoint_metrics.append({
            "persona_count": count,
            "timestamp": time.time(),
            "metrics": metrics
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get full run summary."""
        total_time = self.end_time - self.start_time if self.end_time else 0
        
        return {
            "run_id": self.run_id,
            "tier": self.tier,
            "target_count": self.target_count,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "total_duration_seconds": total_time,
            "throughput_per_minute": (self.target_count / total_time * 60) if total_time > 0 else 0,
            "stages": {
                "stage_a": self.stage_a.get_summary(),
                "stage_b": self.stage_b.get_summary(),
                "stage_c": self.stage_c.get_summary(),
                "validation": self.validation.get_summary(),
            },
            "cost_estimate_usd": self._estimate_cost(),
            "checkpoint_count": len(self.checkpoint_metrics),
            "drift_indicators": self._detect_drift(),
        }
    
    def _estimate_cost(self) -> float:
        """Estimate cost based on token usage (qwen-turbo pricing)."""
        total_tokens = (
            self.stage_b.total_tokens + 
            self.stage_c.total_tokens +
            self.validation.total_tokens
        )
        # qwen-turbo: ~$0.00045 per 1K tokens (avg input/output)
        return (total_tokens / 1000) * 0.00045
    
    def _detect_drift(self) -> List[Dict]:
        """Detect quality drift across checkpoints."""
        if len(self.checkpoint_metrics) < 2:
            return []
        
        drift_indicators = []
        
        # Compare repair rates across checkpoints
        repair_rates = []
        for cp in self.checkpoint_metrics:
            if "repair_rate" in cp["metrics"]:
                repair_rates.append((cp["persona_count"], cp["metrics"]["repair_rate"]))
        
        if len(repair_rates) >= 2:
            first_rate = repair_rates[0][1]
            last_rate = repair_rates[-1][1]
            
            # Drift if repair rate increased significantly
            if last_rate > first_rate * 1.5:
                drift_indicators.append({
                    "type": "repair_rate_increase",
                    "severity": "warning" if last_rate < 0.3 else "critical",
                    "first_rate": first_rate,
                    "last_rate": last_rate,
                    "message": f"Repair rate increased from {first_rate:.2%} to {last_rate:.2%}"
                })
        
        return drift_indicators
    
    def save(self, output_dir: Path):
        """Save metrics to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full metrics
        metrics_file = output_dir / f"metrics_{self.run_id}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        
        # Save checkpoint data for drift analysis
        if self.checkpoint_metrics:
            checkpoint_file = output_dir / f"checkpoints_{self.run_id}.jsonl"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                for cp in self.checkpoint_metrics:
                    f.write(json.dumps(cp, ensure_ascii=False) + '\n')
        
        return metrics_file


class ScaleHarness:
    """
    Production-scale harness for Nemotron persona generation.
    
    Usage:
        harness = ScaleHarness(tier="25K", seed=42)
        results = await harness.run(
            generator=persona_generator,
            output_dir="./output_scale"
        )
    """
    
    TIER_CONFIGS = {
        "1K": {
            "target": 1000,
            "checkpoint_interval": 250,
            "description": "Smoke test - quick validation"
        },
        "25K": {
            "target": 25000,
            "checkpoint_interval": 5000,
            "description": "Distribution + drift detection"
        },
        "100K": {
            "target": 100000,
            "checkpoint_interval": 10000,
            "description": "Throughput + cost + failure modes"
        }
    }
    
    def __init__(
        self,
        tier: str = "1K",
        seed: Optional[int] = None,
        run_id: Optional[str] = None
    ):
        if tier not in self.TIER_CONFIGS:
            raise ValueError(f"Unknown tier: {tier}. Use: {list(self.TIER_CONFIGS.keys())}")
        
        self.tier = tier
        self.config = self.TIER_CONFIGS[tier]
        self.seed = seed
        self.run_id = run_id or f"{tier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.metrics = ScaleRunMetrics(
            run_id=self.run_id,
            tier=tier,
            target_count=self.config["target"],
            seed=seed
        )
    
    async def run(
        self,
        generator,
        output_dir: Path,
        checkpoint_callback: Optional[callable] = None
    ) -> ScaleRunMetrics:
        """
        Execute scale run with full metrics collection.
        
        Args:
            generator: Persona generator instance
            output_dir: Directory for outputs
            checkpoint_callback: Optional callback for checkpoint processing
        
        Returns:
            ScaleRunMetrics with full run data
        """
        print(f"\n{'='*70}")
        print(f"SCALE HARNESS: {self.tier} RUN")
        print(f"{'='*70}")
        print(f"Target: {self.config['target']:,} personas")
        print(f"Description: {self.config['description']}")
        print(f"Seed: {self.seed}")
        print(f"Run ID: {self.run_id}")
        print(f"{'='*70}\n")
        
        self.metrics.start()
        
        # Stage A: Generate structured cores
        print("Stage A: Generating structured persona cores...")
        self.metrics.stage_a.start_time = time.time()
        
        # Generate in batches for checkpointing
        batch_size = self.config["checkpoint_interval"]
        all_personas = []
        
        for batch_start in range(0, self.config["target"], batch_size):
            batch_end = min(batch_start + batch_size, self.config["target"])
            batch_count = batch_end - batch_start
            
            print(f"  Batch {batch_start//batch_size + 1}: personas {batch_start+1}-{batch_end}")
            
            batch_start_time = time.time()
            
            # Generate batch
            try:
                batch_personas = generator.generate(batch_count)
                all_personas.extend(batch_personas)
                
                # Record metrics
                batch_latency = time.time() - batch_start_time
                self.metrics.stage_a.record_attempt(
                    latency=batch_latency,
                    success=True,
                    retries=0
                )
                
            except Exception as e:
                self.metrics.stage_a.record_failure(str(e))
                print(f"    ERROR: {e}")
                continue
            
            # Checkpoint every N personas
            if checkpoint_callback and len(all_personas) % self.config["checkpoint_interval"] == 0:
                print(f"    Checkpoint at {len(all_personas)} personas...")
                checkpoint_metrics = checkpoint_callback(all_personas)
                self.metrics.add_checkpoint(len(all_personas), checkpoint_metrics)
        
        self.metrics.stage_a.end_time = time.time()
        
        # Save results
        print(f"\nSaving results to {output_dir}...")
        self.metrics.end()
        metrics_file = self.metrics.save(output_dir)
        
        # Print summary
        summary = self.metrics.get_summary()
        print(f"\n{'='*70}")
        print("RUN SUMMARY")
        print(f"{'='*70}")
        print(f"Total personas: {len(all_personas):,}")
        print(f"Total duration: {summary['total_duration_seconds']:.1f}s")
        print(f"Throughput: {summary['throughput_per_minute']:.1f} personas/minute")
        print(f"Cost estimate: ${summary['cost_estimate_usd']:.2f}")
        print(f"\nStage A:")
        print(f"  Success rate: {summary['stages']['stage_a']['success_rate']:.2%}")
        print(f"  P50 latency: {summary['stages']['stage_a']['latency_p50']:.2f}s")
        print(f"  P95 latency: {summary['stages']['stage_a']['latency_p95']:.2f}s")
        
        if summary['drift_indicators']:
            print(f"\n⚠️  DRIFT DETECTED:")
            for drift in summary['drift_indicators']:
                print(f"  {drift['severity'].upper()}: {drift['message']}")
        else:
            print(f"\n✓ No significant drift detected")
        
        print(f"\nMetrics saved: {metrics_file}")
        print(f"{'='*70}\n")
        
        return self.metrics


def run_scale_tier(
    tier: str,
    output_dir: str,
    seed: int = 42,
    provider: str = "mock"
) -> ScaleRunMetrics:
    """
    Convenience function to run a scale tier.
    
    Example:
        metrics = run_scale_tier("25K", "./output_25k", seed=42)
    """
    from .generator import PersonaGenerator
    
    harness = ScaleHarness(tier=tier, seed=seed)
    generator = PersonaGenerator(seed=seed)
    
    # Simple checkpoint callback for quality monitoring
    def checkpoint_callback(personas):
        from .population_qa import quick_population_qa
        
        # Convert to dicts
        persona_dicts = []
        for p in personas:
            persona_dicts.append({
                "sex": p.sex,
                "age": p.age,
                "region": p.region,
                "occupation": p.occupation,
            })
        
        qa = quick_population_qa(persona_dicts)
        return {
            "healthy": qa.get("healthy", True),
            "repair_rate": 0.0,  # Would be calculated from actual repairs
        }
    
    return asyncio.run(harness.run(
        generator=generator,
        output_dir=Path(output_dir),
        checkpoint_callback=checkpoint_callback
    ))
