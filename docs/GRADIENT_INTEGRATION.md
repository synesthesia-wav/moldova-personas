# Gradient Integration Guide

**Version:** 1.0.0  
**Purpose:** Single-entrypoint integration for Gradient computed-trust pipeline

---

## Quick Start

```python
from moldova_personas import generate_dataset, UseCaseProfile

# Generate dataset with full trust validation
bundle = generate_dataset(
    n=10000,
    profile=UseCaseProfile.HIGH_STAKES,
    seed=42,
    use_ipf=True,
    strict=True,  # Fail-fast on any issues
)

# Check decision
if bundle.decision == "PASS":
    bundle.save("./output")
else:
    print(f"Quality gates failed: {bundle.decision_reasons}")
    print(f"Gate codes: {[c.value for c in bundle.gate_codes]}")
```

---

## Unified Entrypoint: `generate_dataset()`

### Signature

```python
def generate_dataset(
    n: int,                                    # Number of personas
    profile: UseCaseProfile,                   # ANALYSIS_ONLY, NARRATIVE_REQUIRED, HIGH_STAKES
    seed: Optional[int] = None,                # Random seed
    use_ipf: bool = True,                      # Apply IPF correction
    generate_narratives: bool = False,         # LLM narrative generation
    llm_provider: str = "mock",                # "mock", "openai", "dashscope"
    llm_api_key: Optional[str] = None,         # API key
    llm_model: Optional[str] = None,           # Model name
    strict: Optional[bool] = None,             # Fail-fast (default: from profile)
    outputs: List[str] = None,                 # Output formats
    config_override: Dict[str, Any] = None,    # Additional config
) -> DatasetBundle
```

### Returns: `DatasetBundle`

The bundle contains everything needed for downstream processing:

| Attribute | Type | Description |
|-----------|------|-------------|
| `personas` | `List[Persona]` | Generated persona objects |
| `trust_report` | `TrustReport` | Complete quality metrics |
| `gate_codes` | `List[GateCode]` | Machine-readable gate conditions |
| `decision` | `"PASS" \| "PASS_WITH_WARNINGS" \| "REJECT"` | Final gate decision |
| `decision_reasons` | `List[str]` | Human-readable reasons |
| `run_manifest` | `RunManifest` | Complete reproducibility metadata |
| `ipf_metrics` | `Optional[IPFMetrics]` | ESS, information loss |
| `quality_tier` | `QualityTier` | A/B/C/REJECT |
| `should_escalate` | `bool` | True if human review needed |
| `escalation_priority` | `int` | 1 (highest) to 10 (lowest) |

---

## Machine-Readable Gate Codes

Gate codes provide stable identifiers for automated routing:

### Data Quality
| Code | Severity | Description |
|------|----------|-------------|
| `MARGINAL_ERROR_HIGH` | Warning | Mean L1 error exceeds threshold |
| `MARGINAL_ERROR_CRITICAL` | **Hard Gate** | Maximum category error exceeded |
| `FALLBACK_CRITICAL_FIELD` | Warning | Critical field using fallback |
| `FALLBACK_SUPERCRITICAL_FIELD` | **Hard Gate** | Super-critical field (ethnicity, education, employment, region) using fallback |
| `PXWEB_CACHE_STALE` | Warning | Cached data exceeds max age |
| `PXWEB_FETCH_FAILED` | Warning | Could not fetch from PxWeb |

### IPF / Statistical
| Code | Severity | Description |
|------|----------|-------------|
| `ESS_TOO_LOW` | **Hard Gate** | Effective Sample Size below threshold |
| `WEIGHT_CONCENTRATION_HIGH` | Warning | Some personas over-represented |
| `IPF_DIVERGENCE` | **Hard Gate** | IPF did not converge |

### Narrative Quality
| Code | Severity | Description |
|------|----------|-------------|
| `NARRATIVE_SCHEMA_INVALID` | **Hard Gate** | Response format doesn't match contract |
| `NARRATIVE_PARSE_FAILURE` | **Hard Gate** | Failed to parse LLM response |
| `NARRATIVE_TOO_SHORT` | Warning | Content below minimum length |
| `NARRATIVE_TOO_LONG` | Warning | Content exceeds maximum length |
| `NARRATIVE_LANGUAGE_INVALID` | Warning | Missing Romanian diacritics |
| `NARRATIVE_PRONOUN_MISMATCH` | **Hard Gate** | Pronouns don't match persona sex |
| `NARRATIVE_MOCK_RATIO_HIGH` | **Hard Gate** | Too many empty/mock narratives |
| `NARRATIVE_PII_DETECTED` | **Hard Gate** | Potential PII in narratives |

### Success
| Code | Description |
|------|-------------|
| `PASS` | All quality gates passed |
| `PASS_WITH_WARNINGS` | Passed with non-critical warnings |

### Usage in Gradient

```python
# Route based on gate codes
if GateCode.ESS_TOO_LOW in bundle.gate_codes:
    # Re-run with higher oversample factor
    pass

if GateCode.NARRATIVE_PII_DETECTED in bundle.gate_codes:
    # Escalate to human review
    send_to_review_queue(bundle)

# Priority-based escalation
priority = bundle.escalation_priority
if priority <= 3:
    page_oncall_engineer(bundle)
```

---

## Strict Mode

### Profile-Based Defaults

| Profile | Default Strict | Behavior |
|---------|---------------|----------|
| `ANALYSIS_ONLY` | `False` | Warn on issues, annotate provenance |
| `NARRATIVE_REQUIRED` | `False` | Warn on issues, accept mock narratives |
| `HIGH_STAKES` | `True` | Fail-fast on any schema/config issues |

### Explicit Override

```python
# High-stakes but lenient (for exploration)
bundle = generate_dataset(
    n=1000,
    profile=UseCaseProfile.HIGH_STAKES,
    strict=False,  # Override default
)

# Analysis but strict (for publication)
bundle = generate_dataset(
    n=100000,
    profile=UseCaseProfile.ANALYSIS_ONLY,
    strict=True,  # Override default
)
```

---

## Gradient Computed Trust Mapping

The `to_gradient_trust_payload()` method provides structured data for Gradient's trust system:

```python
payload = bundle.to_gradient_trust_payload()

{
    "decision": "PASS",  # Gating signal
    "quality_tier": "A",
    "confidence": "high",
    "confidence_factors": ["ESS ratio above threshold", "Live PxWeb data"],
    
    "gate_codes": ["PASS"],
    "hard_gates_triggered": false,
    "escalation_priority": 10,
    
    # Computed trust inputs
    "signals": {
        "provenance_coverage_critical": {"PXWEB_DIRECT": 0.3, "CENSUS_HARDCODED": 0.7},
        "targets_source": "mixed",
        "pxweb_cache_age_days": 5,
        "ess_ratio": 0.87,
        "information_loss": 0.13,
        "mean_l1_error": 0.0234,
        "fallback_ratio_super_critical": 0.25,
        "narrative_mock_ratio": 0.0,
    },
    
    "run_id": "20260130_143022",
    "config_hash": "abc123...",
    "generation_time_seconds": 45.2,
}
```

### Mapping to Gradient Trust Concepts

| Gradient Concept | Bundle Source |
|-----------------|---------------|
| **Gating** | `decision` (PASS / PASS_WITH_WARNINGS / REJECT) |
| **Confidence** | `trust_report.trust_decision.confidence` |
| **Explainability** | `confidence_factors` + `gate_codes` |
| **Computed Trust Inputs** | `signals` dict (provenance, ESS, cache age, etc.) |
| **Escalation Routing** | `should_escalate` + `escalation_priority` |
| **Audit Trail** | `run_id` + `config_hash` |

---

## API Compatibility Guarantee

The public API surface is locked and tested:

### Exported Symbols (Stable)

```python
from moldova_personas import (
    # Core generation
    PersonaGenerator, Persona,
    
    # Unified entrypoint
    generate_dataset, DatasetBundle, DatasetConfig,
    
    # Trust system
    TrustReportGenerator, TrustReport, TrustDecisionRecord,
    QualityTier, UseCaseProfile, GateCode,
    
    # Reproducibility
    RunManifest, create_run_manifest_from_trust_report,
    
    # Export
    export_all_formats,
)
```

### Version Compatibility

- **Major (X.0.0)**: Breaking changes to `generate_dataset()` signature or `DatasetBundle` structure
- **Minor (x.Y.0)**: New fields added to bundles, new gate codes
- **Patch (x.y.Z)**: Bug fixes, documentation

### Compatibility Tests

The test `tests/test_api_compatibility.py` verifies:
- All required symbols are exported
- Function signatures haven't changed
- Enum values are stable
- Backward compatibility

---

## Operational Playbook

### When to Use Each Profile

| Use Case | Profile | Strict | Notes |
|----------|---------|--------|-------|
| Exploration, prototyping | `ANALYSIS_ONLY` | `False` | Fast, no LLM costs |
| Internal analysis | `ANALYSIS_ONLY` | `True` | Quality for decisions |
| Marketing, content | `NARRATIVE_REQUIRED` | `False` | Accept some mock narratives |
| Public dataset | `HIGH_STAKES` | `True` | Full validation required |
| Academic publication | `HIGH_STAKES` | `True` | Document everything |

### Handling Rejections

```python
bundle = generate_dataset(n=10000, profile=UseCaseProfile.HIGH_STAKES)

if bundle.decision == "REJECT":
    # Analyze gate codes
    codes = {c.value for c in bundle.gate_codes}
    
    if GateCode.ESS_TOO_LOW in codes:
        # Re-run with higher oversample
        bundle = generate_dataset(
            n=10000, 
            profile=UseCaseProfile.HIGH_STAKES,
            config_override={"oversample_factor": 5}
        )
    
    elif GateCode.FALLBACK_SUPERCRITICAL_FIELD in codes:
        # Force PxWeb refresh
        from moldova_personas import NBSDataManager
        NBSDataManager().refresh_cache()
        bundle = generate_dataset(n=10000, profile=UseCaseProfile.HIGH_STAKES)
    
    elif GateCode.NARRATIVE_SCHEMA_INVALID in codes:
        # Check LLM provider status
        # May need to switch provider or accept mock narratives
        bundle = generate_dataset(
            n=10000,
            profile=UseCaseProfile.HIGH_STAKES,
            generate_narratives=False,  # Skip narratives
        )
```

---

## Performance Expectations

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Raw generation | ~50k/sec | Structured data only |
| With IPF | ~17k/sec | 3× oversample + resampling |
| With narratives (mock) | ~50k/sec | Empty strings, no cost |
| With narratives (GPT-3.5) | ~1000/min | Parallel workers |
| Export (Parquet) | ~125k/sec | All formats |
| Trust validation | ~100k/sec | Post-generation |

---

## Integration Checklist

- [ ] Import `generate_dataset` and `UseCaseProfile`
- [ ] Choose profile based on use case
- [ ] Set `strict=True` for production
- [ ] Handle `decision == "REJECT"` case
- [ ] Route on `gate_codes` for escalation
- [ ] Use `to_gradient_trust_payload()` for trust signals
- [ ] Store `run_manifest` for reproducibility
- [ ] Archive `trust_report` for audit

---

## Support

- **Issues:** GitHub Issues with `run_manifest.json`
- **Gate Code Reference:** This document
- **Compatibility:** `tests/test_api_compatibility.py`
