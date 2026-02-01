# Minimal Metrics Specification

Production logging format for Nemotron-scale persona generation.

## Design Principles

1. **One log line per generation attempt** (machine parsable)
2. **Aggregatable at any level** (persona, batch, tier)
3. **Tells you in 25K run if "feel" is collapsing**

---

## Log Schema (per attempt)

```json
{
  "ts": "2026-01-31T14:32:01.234Z",
  "run_id": "25K_20260131_143200",
  "persona_id": "uuid-or-sequence",
  "stage": "B|C|validation",
  "attempt": 1,
  "status": "success|fail|repair",
  
  "latency_ms": 1250,
  "prompt_tokens": 450,
  "completion_tokens": 320,
  
  "repairs": ["trait_leak", "opening_repetition"],
  "validator_scores": {
    "trait_leak": 0,
    "opening_variation": 1,
    "pollyanna": 0.7,
    "anchor_specificity": 0.8
  },
  
  "opening_move": "situational_hook",
  "constraint_categories": ["time_pressure", "family_duty"],
  "anchors_used": ["piața din Cahul", "zacuscă"],
  
  "error": null
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `ts` | ISO8601 | Timestamp |
| `run_id` | string | Unique run identifier |
| `persona_id` | string | UUID or sequence number |
| `stage` | enum | Generation stage |
| `attempt` | int | Retry count (1=first) |
| `status` | enum | Final status |
| `latency_ms` | int | Wall time in ms |
| `prompt_tokens` | int | Input tokens |
| `completion_tokens` | int | Output tokens |
| `repairs` | array | Types of repairs applied |
| `validator_scores` | object | Per-validator 0-1 scores |
| `opening_move` | string | Rhetorical move used |
| `constraint_categories` | array | Which constraints applied |
| `anchors_used` | array | Which anchors appeared |
| `error` | string\|null | Error message if failed |

---

## Aggregation Rules

### Per-Batch Metrics (every N personas)

```python
def aggregate_batch(logs: List[Log]) -> Dict:
    return {
        "batch_size": len(logs),
        "success_rate": sum(1 for l in logs if l.status == "success") / len(logs),
        "mean_latency_ms": statistics.mean(l.latency_ms for l in logs),
        "p95_latency_ms": percentile(l.latency_ms for l in logs, 95),
        
        # Repair rates (early warning for collapse)
        "repair_rate": sum(1 for l in logs if l.repairs) / len(logs),
        "repair_breakdown": Counter(
            r for l in logs for r in l.repairs
        ),
        
        # Quality indicators
        "mean_validator_scores": {
            v: statistics.mean(l.validator_scores[v] for l in logs)
            for v in ["trait_leak", "opening_variation", "pollyanna"]
        },
        
        # Diversity indicators
        "opening_move_entropy": entropy(
            Counter(l.opening_move for l in logs).values()
        ),
        "anchor_diversity": len(set(
            a for l in logs for a in l.anchors_used
        )) / len(logs),
        
        # Cost
        "total_tokens": sum(l.prompt_tokens + l.completion_tokens for l in logs),
        "cost_usd": total_tokens * 0.00045 / 1000,
    }
```

### Per-Tier Summary Metrics

```python
def aggregate_tier(batches: List[BatchMetrics]) -> Dict:
    return {
        "tier": "25K",
        "total_personas": sum(b.batch_size for b in batches),
        "overall_success_rate": weighted_average(
            b.success_rate for b in batches
        ),
        
        # Drift detection
        "repair_rate_trend": slope(
            b.repair_rate for b in batches
        ),
        "latency_trend": slope(
            b.mean_latency_ms for b in batches
        ),
        
        # Quality trends
        "opening_entropy_trend": slope(
            b.opening_move_entropy for b in batches
        ),
        "pollyanna_score_trend": slope(
            b.mean_validator_scores["pollyanna"] for b in batches
        ),
        
        # Drift alerts
        "drift_detected": any([
            repair_rate_trend > 0.5,  # Repairs increasing
            opening_entropy_trend < -0.1,  # Variety decreasing
            pollyanna_score_trend > 0.2,  # Getting too positive
        ]),
    }
```

---

## Key Indicators of "Feel" Collapse

| Indicator | Healthy | Warning | Critical |
|-----------|---------|---------|----------|
| Repair rate | <10% | 10-20% | >20% |
| Repair rate trend | Stable | +5%/batch | +10%/batch |
| Opening entropy | >2.0 | 1.5-2.0 | <1.5 |
| Pollyanna score | 0.5-0.7 | 0.7-0.85 | >0.85 |
| Anchor diversity | >0.8 | 0.6-0.8 | <0.6 |
| Mean latency | Stable | +20% | +50% |

---

## Quick Implementation

```python
# Logger setup
import json
import logging

logger = logging.getLogger("nemotron.generation")
handler = logging.FileHandler("generation.log")
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

def log_attempt(log_entry: dict):
    """Write single log line."""
    logger.info(json.dumps(log_entry, ensure_ascii=False))

# Example usage
log_attempt({
    "ts": datetime.now().isoformat(),
    "run_id": run_id,
    "persona_id": persona.uuid,
    "stage": "B",
    "attempt": 2,
    "status": "success",
    "latency_ms": 1234,
    "prompt_tokens": 450,
    "completion_tokens": 320,
    "repairs": ["trait_leak"],
    "validator_scores": {
        "trait_leak": 0,
        "opening_variation": 0.8,
        "pollyanna": 0.6,
    },
    "opening_move": "situational_hook",
    "constraint_categories": ["time_pressure"],
    "anchors_used": ["piața din Cahul"],
    "error": None,
})
```

---

## Analysis Commands

```bash
# Extract repair rates by batch
cat generation.log | \
  jq -s 'group_by(.persona_id | split("-")[0]) | 
         map({batch: .[0].persona_id, 
              repair_rate: map(select(.repairs | length > 0)) | length / length})'

# Check for drift
cat generation.log | \
  jq -s 'group_by(.persona_id | tonumber / 1000 | floor) | 
         map({batch: .[0].persona_id, 
              mean_pol: map(.validator_scores.pollyanna) | add / length})'

# Cost estimation
cat generation.log | \
  jq -s 'map(.prompt_tokens + .completion_tokens) | add | . * 0.00045 / 1000'
```

---

## Dashboard Integration

Feed logs to:

1. **Prometheus/Grafana** - Real-time metrics
2. **ELK Stack** - Log analysis
3. **BigQuery** - Large-scale analytics
4. **Custom dashboard** - Quality monitoring

---

## Files

| File | Purpose |
|------|---------|
| `metrics_collector.py` | Log writing utilities |
| `metrics_aggregator.py` | Batch/tier aggregation |
| `metrics_dashboard.py` | Visualization/reporting |
