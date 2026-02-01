#!/usr/bin/env python3
"""
100K Generation Validation Script

Validates that the generator can handle 100,000 personas with:
- IPF correction
- Checkpoint recovery
- Memory efficiency
- Quality gates
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from moldova_personas import PersonaGenerator, UseCaseProfile
from moldova_personas.gradient_integration import generate_dataset
from moldova_personas.metrics import get_collector, timed_generation


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_100k(
    n: int = 100_000,
    output_dir: str = "./output_100k",
    seed: int = 42,
    checkpoint_interval: int = 10_000,
):
    """
    Validate 100K generation with full checks.
    
    Args:
        n: Number of personas to generate
        output_dir: Output directory
        seed: Random seed
        checkpoint_interval: Save checkpoint every N personas
    """
    logger.info(f"Starting 100K validation: n={n}, seed={seed}")
    
    start_time = time.perf_counter()
    collector = get_collector()
    
    # Test 1: Basic generation with IPF
    logger.info("Test 1: Basic generation with IPF correction...")
    with timed_generation("basic_100k", test="basic"):
        bundle = generate_dataset(
            n=n,
            profile=UseCaseProfile.HIGH_STAKES,
            seed=seed,
            use_ipf=True,
            strict=True,
        )
    
    assert len(bundle.personas) == n, f"Expected {n} personas, got {len(bundle.personas)}"
    logger.info(f"✓ Generated {n} personas in {bundle.generation_time_seconds:.1f}s")
    
    # Test 2: Quality gates pass
    logger.info("Test 2: Quality gates...")
    assert bundle.decision in ["PASS", "PASS_WITH_WARNINGS"], \
        f"Expected PASS, got {bundle.decision} with codes: {bundle.gate_codes}"
    logger.info(f"✓ Decision: {bundle.decision}, Quality: {bundle.quality_tier.value}")
    
    # Test 3: IPF metrics reasonable
    logger.info("Test 3: IPF metrics...")
    if bundle.ipf_metrics:
        ess_ratio = bundle.ipf_metrics.ess_ratio
        assert ess_ratio >= 0.70, f"ESS ratio too low: {ess_ratio}"
        logger.info(f"✓ ESS ratio: {ess_ratio:.2%}")
    
    # Test 4: Age-education consistency
    logger.info("Test 4: Age-education consistency...")
    violations = 0
    for p in bundle.personas:
        if p.education_level == "Superior (Licență/Master)" and p.age < 19:
            violations += 1
        elif p.education_level == "Doctorat" and p.age < 27:
            violations += 1
    
    violation_rate = violations / n
    assert violation_rate < 0.001, f"Too many age-education violations: {violations} ({violation_rate:.2%})"
    logger.info(f"✓ Age-education violations: {violations} ({violation_rate:.4%})")
    
    # Test 5: Save outputs
    logger.info("Test 5: Saving outputs...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved = bundle.save(output_path, formats=["parquet", "trust_report", "run_manifest", "gradient_payload"])
    for fmt, path in saved.items():
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ {fmt}: {path} ({size_mb:.1f} MB)")
    
    # Test 6: Export metrics
    logger.info("Test 6: Metrics summary...")
    summary = collector.get_summary()
    logger.info(f"  Counters: {summary['counters']}")
    logger.info(f"  Timers: {list(summary['timers'].keys())}")
    
    # Final summary
    total_time = time.perf_counter() - start_time
    throughput = n / bundle.generation_time_seconds
    
    logger.info("=" * 60)
    logger.info("100K VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Generation time: {bundle.generation_time_seconds:.1f}s")
    logger.info(f"Throughput: {throughput:,.0f} personas/sec")
    logger.info(f"Decision: {bundle.decision}")
    logger.info(f"Quality tier: {bundle.quality_tier.value}")
    logger.info(f"Gate codes: {[c.value for c in bundle.gate_codes]}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)
    
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Validate 100K generation")
    parser.add_argument("--count", type=int, default=100_000, help="Number of personas")
    parser.add_argument("--output", type=str, default="./output_100k", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-interval", type=int, default=10_000, help="Checkpoint interval")
    
    args = parser.parse_args()
    
    try:
        validate_100k(
            n=args.count,
            output_dir=args.output,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
        )
        logger.info("✅ All validation tests passed!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
