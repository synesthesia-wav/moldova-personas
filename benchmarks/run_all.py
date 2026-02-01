"""
Run all benchmarks and generate comprehensive report.
"""

import subprocess
import json
import sys
from datetime import datetime
from pathlib import Path


BENCHMARKS = [
    ("Generation (Raw)", "bench_generation.py", ["--count", "10000"]),
    ("Generation (IPF)", "bench_generation.py", ["--count", "10000", "--ipf"]),
]


def run_benchmark(name: str, script: str, args: list) -> dict:
    """Run a single benchmark and parse results."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script] + args + ["--format", "json"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return {"name": name, "status": "error", "error": result.stderr}
        
        # Parse JSON output (last line should be JSON)
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            try:
                data = json.loads(line)
                data["name"] = name
                data["status"] = "success"
                return data
            except json.JSONDecodeError:
                continue
        
        return {"name": name, "status": "error", "error": "No JSON output found"}
        
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "timeout"}
    except Exception as e:
        return {"name": name, "status": "error", "error": str(e)}


def main():
    print("Moldova Personas Generator - Benchmark Suite")
    print("=" * 60)
    
    all_results = {
        "suite_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "benchmarks": [],
    }
    
    for name, script, args in BENCHMARKS:
        result = run_benchmark(name, script, args)
        all_results["benchmarks"].append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for bench in all_results["benchmarks"]:
        status = bench.get("status", "unknown")
        name = bench["name"]
        
        if status == "success":
            for test in bench.get("tests", []):
                throughput = test.get("throughput", 0)
                print(f"✓ {name}: {throughput:,.0f} personas/sec")
        elif status == "timeout":
            print(f"⏱ {name}: TIMEOUT")
        else:
            print(f"✗ {name}: ERROR")
    
    # Save full results
    results_file = Path(__file__).parent / "results" / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nFull results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
