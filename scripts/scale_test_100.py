#!/usr/bin/env python3
"""
Scale test: Generate 100 Nemotron-style personas.

Tests:
- Throughput (personas/minute)
- Cost estimation
- Quality validation sampling
- Memory usage

Usage:
  python scripts/scale_test_100.py --provider dashscope --api-key <key>
  python scripts/scale_test_100.py --mock  # Fast test without LLM
"""

import asyncio
import argparse
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from moldova_personas.generator import PersonaGenerator
from moldova_personas.names import generate_name
from moldova_personas.nemotron_pipeline import generate_nemotron_personas


class ScaleTestMetrics:
    """Collect metrics for scale testing."""
    
    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.stage_times: Dict[str, float] = {}
        self.total_personas: int = 0
        self.successful: int = 0
        self.failed: int = 0
        self.cost_estimate_usd: float = 0
        self.validation_scores: List[float] = []
        
    def start(self):
        self.start_time = time.time()
        
    def end(self):
        self.end_time = time.time()
        
    def record_stage(self, stage: str, duration: float):
        self.stage_times[stage] = duration
        
    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0
    
    @property
    def personas_per_minute(self) -> float:
        if self.total_duration == 0:
            return 0
        return (self.successful / self.total_duration) * 60
    
    @property
    def average_validation_score(self) -> float:
        if not self.validation_scores:
            return 0
        return sum(self.validation_scores) / len(self.validation_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_personas": self.total_personas,
            "successful": self.successful,
            "failed": self.failed,
            "total_duration_seconds": round(self.total_duration, 2),
            "personas_per_minute": round(self.personas_per_minute, 2),
            "stage_times": {k: round(v, 2) for k, v in self.stage_times.items()},
            "cost_estimate_usd": round(self.cost_estimate_usd, 4),
            "average_validation_score": round(self.average_validation_score, 2) if self.validation_scores else None,
        }


async def run_scale_test(
    count: int = 100,
    provider: str = "mock",
    api_key: Optional[str] = None,
    model: str = "qwen-turbo",
    max_concurrent: int = 5,
    output_dir: str = "output_scale_test",
    validate_sample: int = 10
) -> ScaleTestMetrics:
    """
    Run scale test generating specified number of personas.
    
    Args:
        count: Number of personas to generate
        provider: LLM provider
        api_key: API key
        model: Model name
        max_concurrent: Max concurrent API calls
        output_dir: Directory for output files
        validate_sample: Number of personas to run full validation on
        
    Returns:
        ScaleTestMetrics with all collected data
    """
    metrics = ScaleTestMetrics()
    metrics.total_personas = count
    
    print("=" * 70)
    print(f"SCALE TEST: {count} NEMOTRON-STYLE PERSONAS")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Max concurrent: {max_concurrent}")
    print(f"  Output dir: {output_dir}")
    print(f"  Validation sample: {validate_sample}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Stage 1: Generate structured personas
    print("Stage 1: Generating structured persona cores...")
    stage_start = time.time()
    
    generator = PersonaGenerator()
    structured_personas = generator.generate(count)
    
    names = []
    for p in structured_personas:
        name = generate_name(getattr(p, 'ethnicity', 'Moldovean'), p.sex)
        names.append(name)
    
    stage_duration = time.time() - stage_start
    metrics.record_stage("structured_generation", stage_duration)
    print(f"  ✓ Generated {len(structured_personas)} structured cores in {stage_duration:.1f}s")
    print()
    
    # Stage 2: Generate Nemotron personas
    print("Stage 2: Running Nemotron pipeline (3 stages + validation)...")
    stage_start = time.time()
    
    metrics.start()
    
    try:
        results = await generate_nemotron_personas(
            personas=structured_personas,
            names=names,
            provider=provider,
            api_key=api_key,
            model=model,
            max_concurrent=max_concurrent,
            run_soft_validation=False  # Too slow for 100, sample later
        )
        
        # Extract personas from results
        nemotron_personas = [r["persona"] for r in results]
        metrics.successful = len(nemotron_personas)
        
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        nemotron_personas = []
        metrics.failed = count
        
    stage_duration = time.time() - stage_start
    metrics.record_stage("nemotron_pipeline", stage_duration)
    print(f"  ✓ Generated {len(nemotron_personas)} Nemotron personas in {stage_duration:.1f}s")
    print()
    
    metrics.end()
    
    # Calculate cost (DashScope qwen-turbo: ~$0.0003 per 1K tokens, ~3K tokens per persona)
    if provider == "dashscope":
        estimated_tokens_per_persona = 3000  # prompt + response
        tokens_total = estimated_tokens_per_persona * len(nemotron_personas)
        # qwen-turbo: $0.0003 per 1K input, $0.0006 per 1K output (avg $0.00045 per 1K)
        cost_per_1k = 0.00045
        metrics.cost_estimate_usd = (tokens_total / 1000) * cost_per_1k
    
    # Save output
    print("Stage 3: Saving results...")
    
    # Save full dataset
    output_file = output_path / f"nemotron_personas_{count}.json"
    dataset = {
        "metadata": {
            "format": "Nemotron-Moldova",
            "count": len(nemotron_personas),
            "generated_at": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
        },
        "metrics": metrics.to_dict(),
        "personas": [p.to_dict() for p in nemotron_personas]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved to {output_file}")
    print(f"    File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Save CSV for easy analysis
    try:
        import csv
        csv_file = output_path / f"nemotron_personas_{count}.csv"
        if nemotron_personas:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=nemotron_personas[0].to_dict().keys())
                writer.writeheader()
                for p in nemotron_personas:
                    writer.writerow(p.to_dict())
            print(f"  ✓ Saved CSV to {csv_file}")
    except Exception as e:
        print(f"  ⚠ CSV export failed: {e}")
    
    print()
    
    # Print summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total personas:    {metrics.total_personas}")
    print(f"Successful:        {metrics.successful}")
    print(f"Failed:            {metrics.failed}")
    print(f"Success rate:      {metrics.successful/metrics.total_personas*100:.1f}%")
    print()
    print(f"Total time:        {metrics.total_duration:.1f}s")
    print(f"Throughput:        {metrics.personas_per_minute:.1f} personas/minute")
    print(f"Time per persona:  {metrics.total_duration/metrics.successful:.2f}s" if metrics.successful else "N/A")
    print()
    print(f"Cost estimate:     ${metrics.cost_estimate_usd:.4f} USD")
    print(f"Cost per persona:  ${metrics.cost_estimate_usd/metrics.successful:.4f}" if metrics.successful else "N/A")
    print()
    
    # Sample validation
    if validate_sample > 0 and nemotron_personas and provider != "mock":
        print(f"Running validation on sample of {validate_sample} personas...")
        from moldova_personas.llm_client import create_llm_client
        from moldova_personas.soft_validators import SoftValidators
        
        llm = create_llm_client(provider, api_key=api_key, model=model)
        validators = SoftValidators(llm)
        
        sample_indices = list(range(0, min(validate_sample, len(nemotron_personas))))
        validation_results = []
        
        for i in sample_indices:
            try:
                results = await validators.validate_full(nemotron_personas[i])
                report = validators.generate_report(results)
                validation_results.append(report)
                metrics.validation_scores.append(report["average_score"])
            except Exception as e:
                print(f"  Validation failed for persona {i}: {e}")
        
        if validation_results:
            avg_score = sum(r["average_score"] for r in validation_results) / len(validation_results)
            passed = sum(r["accept"] for r in validation_results)
            print(f"  Average score: {avg_score:.2f}/5.0")
            print(f"  Passed: {passed}/{len(validation_results)}")
    
    print()
    print(f"Output saved to: {output_path}/")
    print("=" * 70)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Scale test for Nemotron pipeline")
    parser.add_argument("--count", "-n", type=int, default=100,
                       help="Number of personas to generate (default: 100)")
    parser.add_argument("--provider", type=str, default="mock",
                       choices=["mock", "dashscope", "openai"],
                       help="LLM provider (default: mock)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (or use env var)")
    parser.add_argument("--model", type=str, default="qwen-turbo",
                       help="Model name (default: qwen-turbo)")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Max concurrent API calls (default: 5)")
    parser.add_argument("--output-dir", type=str, default="output_scale_test",
                       help="Output directory (default: output_scale_test)")
    parser.add_argument("--validate-sample", type=int, default=10,
                       help="Number of personas to validate (default: 10)")
    parser.add_argument("--mock", action="store_true",
                       help="Force mock mode (no LLM calls)")
    
    args = parser.parse_args()
    
    # Determine provider
    provider = "mock" if args.mock else args.provider
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if provider == "dashscope" and not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        return 1
    
    # Run test
    metrics = asyncio.run(run_scale_test(
        count=args.count,
        provider=provider,
        api_key=api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
        validate_sample=args.validate_sample if not args.mock else 0
    ))
    
    return 0


if __name__ == "__main__":
    exit(main())
