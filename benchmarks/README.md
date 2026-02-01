# Performance Benchmarks

Reproducible performance benchmarks for the Moldova Personas Generator.

## Environment

Benchmarks should be run on a standardized environment:

- **Python**: 3.10+ (see `pyproject.toml` for exact version constraints)
- **OS**: macOS/Linux (Windows results may vary)
- **Hardware**: Documented per run
- **Dependencies**: Locked in `requirements.txt`

## Running Benchmarks

```bash
# Install dependencies
pip install -r requirements.txt

# Run all benchmarks
python benchmarks/run_all.py

# Run specific benchmark
python benchmarks/bench_generation.py --count 10000

# Run with custom parameters
python benchmarks/bench_generation.py --count 50000 --ipf --output results.json
```

## Benchmark Suite

| Benchmark | Description | Typical Range |
|-----------|-------------|---------------|
| `bench_generation.py` | Structured persona generation | 10k-100k personas/sec |
| `bench_ipf.py` | IPF correction overhead | 2-5x slower than raw |
| `bench_export.py` | Export format performance | Varies by format |
| `bench_narrative.py` | LLM narrative generation | 10-1000 personas/min |

## Interpreting Results

Benchmarks output:
- **Throughput**: personas/second (higher is better)
- **Latency**: milliseconds per persona (lower is better)
- **Memory**: peak RAM usage
- **Scaling**: how performance changes with N

## Historical Results

See `benchmarks/results/` for archived benchmark runs.

### Reference Results (Apple M3 Pro, 2026-01-30)

| Operation | Count | Time | Throughput |
|-----------|-------|------|------------|
| Raw generation | 100,000 | 2.1s | ~48k/sec |
| With IPF | 100,000 | 5.8s | ~17k/sec |
| Export Parquet | 100,000 | 0.8s | ~125k/sec |
| Export JSON | 100,000 | 2.4s | ~42k/sec |
