"""
Benchmark: Structured Persona Generation

Measures throughput and latency of persona generation without LLM narratives.
"""

import argparse
import json
import time
import platform
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from moldova_personas import PersonaGenerator


def benchmark_raw_generation(n: int, seed: int = 42) -> dict:
    """Benchmark raw persona generation (no IPF)."""
    generator = PersonaGenerator(seed=seed)
    
    start = time.perf_counter()
    personas = generator.generate(n, show_progress=False, use_ethnicity_correction=False)
    elapsed = time.perf_counter() - start
    
    return {
        "operation": "raw_generation",
        "count": n,
        "elapsed_seconds": elapsed,
        "throughput": n / elapsed,
        "latency_ms": (elapsed / n) * 1000,
    }


def benchmark_ipf_generation(n: int, oversample_factor: int = 3, seed: int = 42) -> dict:
    """Benchmark persona generation with IPF correction."""
    generator = PersonaGenerator(seed=seed)
    
    start = time.perf_counter()
    personas, metrics = generator.generate_with_ethnicity_correction(
        n, 
        oversample_factor=oversample_factor,
        return_metrics=True
    )
    elapsed = time.perf_counter() - start
    
    return {
        "operation": "ipf_generation",
        "count": n,
        "oversample_factor": oversample_factor,
        "elapsed_seconds": elapsed,
        "throughput": n / elapsed,
        "latency_ms": (elapsed / n) * 1000,
        "ipf_ess_ratio": metrics.ess_ratio,
        "ipf_information_loss": metrics.information_loss,
    }


def get_system_info() -> dict:
    """Get system information for reproducibility."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark persona generation")
    parser.add_argument("--count", type=int, default=10000, help="Number of personas")
    parser.add_argument("--ipf", action="store_true", help="Include IPF correction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--format", choices=["json", "markdown", "text"], default="text")
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = {
        "benchmark": "generation",
        "timestamp": datetime.now().isoformat(),
        "system": get_system_info(),
        "tests": [],
    }
    
    print(f"Benchmarking generation ({args.count} personas)...")
    
    # Raw generation
    raw_result = benchmark_raw_generation(args.count, args.seed)
    results["tests"].append(raw_result)
    
    if args.ipf:
        print(f"Benchmarking with IPF...")
        ipf_result = benchmark_ipf_generation(args.count, seed=args.seed)
        results["tests"].append(ipf_result)
    
    # Output results
    if args.format == "json":
        output = json.dumps(results, indent=2)
    elif args.format == "markdown":
        lines = [
            "# Generation Benchmark Results",
            f"**Date:** {results['timestamp']}",
            f"**Platform:** {results['system']['platform']}",
            f"**Python:** {results['system']['python_version']}",
            "",
            "| Operation | Count | Time (s) | Throughput | Latency (ms) |",
            "|-----------|-------|----------|------------|--------------|",
        ]
        for test in results["tests"]:
            lines.append(
                f"| {test['operation']} | {test['count']:,} | "
                f"{test['elapsed_seconds']:.3f} | "
                f"{test['throughput']:,.0f}/sec | "
                f"{test['latency_ms']:.4f} |"
            )
        output = "\n".join(lines)
    else:
        lines = [
            f"Generation Benchmark",
            f"====================",
            f"Timestamp: {results['timestamp']}",
            f"Platform: {results['system']['platform']}",
            f"Python: {results['system']['python_version']}",
            "",
        ]
        for test in results["tests"]:
            lines.extend([
                f"Operation: {test['operation']}",
                f"  Count: {test['count']:,}",
                f"  Time: {test['elapsed_seconds']:.3f}s",
                f"  Throughput: {test['throughput']:,.0f} personas/sec",
                f"  Latency: {test['latency_ms']:.4f} ms/persona",
            ])
            if "ipf_ess_ratio" in test:
                lines.append(f"  IPF ESS Ratio: {test['ipf_ess_ratio']:.2%}")
            lines.append("")
        output = "\n".join(lines)
    
    print(output)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
