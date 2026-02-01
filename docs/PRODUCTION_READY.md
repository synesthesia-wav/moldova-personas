# Moldova Personas Generator — Production Readiness Guide

**Version:** 1.0.0  
**Last Updated:** 2026-01-30  
**Status:** ✅ Production Ready

---

## 1. What It Is / Non-Goals

### What It Is
A scientifically rigorous synthetic population generator for Moldova, grounded in official 2024 Census data from the National Bureau of Statistics (NBS). Uses Probabilistic Graphical Models (PGM) for structured generation and optional LLM for narrative enrichment.

### Core Capabilities
- **Structured Generation**: Demographics, location, education, occupation (100k+ personas/sec)
- **IPF Correction**: Adjusts ethnicity distribution to match census marginals
- **Provenance Tracking**: Every field has `DataProvenance` (PxWeb, IPF-derived, hardcoded)
- **Trust Gating**: Quality tiers (A/B/C/REJECT) with use-case profiles
- **Run Manifest**: Complete reproducibility artifact (config hash, data sources, IPF metrics)

### Non-Goals
- **Not** a real-time API service (batch generation only)
- **Not** a general-purpose synthetic data tool (Moldova-specific)
- **Not** a replacement for actual census microdata (statistical aggregate only)
- **Not** guaranteed to produce narratives without LLM costs

---

## 2. Data Sources + Provenance Guarantees

### Hierarchy of Trust

| Priority | Source | Confidence | Fields |
|----------|--------|------------|--------|
| 1 | PxWeb API (live) | 0.99 | sex, residence_type, age_group |
| 2 | PxWeb (cached <30d) | 0.95 | sex, residence_type, age_group |
| 3 | IPF-derived | 0.85 | ethnicity×region, region×urban |
| 4 | NBS 2024 hardcoded | 0.90-0.95 | ethnicity, religion, education, marital |
| 5 | Documented estimates | 0.70-0.80 | occupation sector, ethnicity-religion |

### Data Currency
- **Cache TTL**: 30 days (configurable)
- **Reference Year**: 2024 (NBS Census)
- **Refresh**: Manual via `NBSDataManager.refresh_cache()`

### Schema Drift Protection
- **Golden Fixtures**: `tests/fixtures/pxweb/` for contract testing
- **Category Alignment**: `CategoryMapping` manifest validates external→internal labels
- **Network Tests**: Optional `@pytest.mark.network` tests validate live API

---

## 3. Generation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Target count N, seed, use_case_profile              │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Structured Generation (PGM)                       │
│  - Sample from census distributions                         │
│  - Enforce dependency graph (Region → Ethnicity → Name)     │
│  - Age-appropriate education constraints                    │
│  Throughput: ~50k personas/sec                              │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: IPF Correction (Optional)                         │
│  - Oversample 3×                                            │
│  - Weight by ethnicity target/actual                        │
│  - Resample to target N                                     │
│  - Track ESS, weight_concentration, info_loss               │
│  Overhead: ~2-3× slower                                     │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Narrative Enrichment (Optional)                   │
│  - Generate Romanian prompts                                │
│  - LLM call (OpenAI/DashScope/local)                        │
│  - Parse sections, validate contract                        │
│  - Extract skills/hobbies                                   │
│  Throughput: 10-1000/min (depends on provider)              │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Trust Validation                                  │
│  - Compare marginals to census targets                      │
│  - Calculate L1 error, ESS ratio                            │
│  - Apply use-case gating                                    │
│  - Generate TrustReport + RunManifest                       │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Artifact Bundle                                    │
│  - moldova_personas.parquet (or json/jsonl/csv)             │
│  - trust_report.json                                        │
│  - run_manifest.json                                        │
│  - stats.md (human-readable)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Trust Gating Profiles

### UseCaseProfile.ANALYSIS_ONLY
**For:** Statistical analysis, no narratives needed
```python
max_mean_l1_error = 0.10
max_fallback_ratio_critical = 0.70
min_ess_ratio = 0.50
max_mock_ratio = 1.0  # Narratives irrelevant
```

### UseCaseProfile.NARRATIVE_REQUIRED
**For:** Datasets with LLM-generated text (default)
```python
max_mean_l1_error = 0.10
max_fallback_ratio_critical = 0.70
min_ess_ratio = 0.70  # Gate if <70%
max_mock_ratio = 0.20  # Max 20% mock narratives
min_schema_valid_ratio_attempted = 0.90
```

### UseCaseProfile.HIGH_STAKES
**For:** Public release, policy decisions, academic publication
```python
max_mean_l1_error = 0.05
max_fallback_ratio_critical = 0.30
max_fallback_ratio_super_critical = 0.10  # Ethnicity, education, employment, region
min_ess_ratio = 0.85
max_mock_ratio = 0.05
min_schema_valid_ratio_attempted = 0.95
max_cache_age_days = 7  # Very fresh data required
```

### Hard Gate Triggers
A dataset is **REJECT** if any of:
- `mean_l1_error > threshold` for the profile
- `ess_ratio < min_ess_ratio`
- `schema_valid_ratio_attempted < threshold` (narrative modes)
- Any super-critical field uses fallback (HIGH_STAKES only)

---

## 5. Run Artifacts

Every generation produces a complete artifact bundle:

### Required Outputs

| File | Format | Contents |
|------|--------|----------|
| `moldova_personas.parquet` | Apache Parquet | Full persona dataset (recommended) |
| `moldova_personas.jsonl` | JSON Lines | Alternative format |
| `trust_report.json` | JSON | Quality metrics, gating decisions |
| `run_manifest.json` | JSON | Complete reproducibility metadata |
| `stats.md` | Markdown | Human-readable summary |

### Trust Report Structure
```json
{
  "report_id": "20260130_143022",
  "quality_tiers": {"structured": "A", "narrative": "B", "overall": "B"},
  "mean_l1_error": 0.0234,
  "fallback_ratio_critical": 0.30,
  "ipf_metrics": {
    "ess_ratio": 0.87,
    "information_loss": 0.13,
    "weight_concentration": 3.4
  },
  "trust_decision": {
    "decision": "PASS_WITH_WARNINGS",
    "confidence": "high",
    "basis": ["ESS ratio above threshold", "No hard gates triggered"]
  },
  "hard_gate_triggered": false
}
```

### Run Manifest Structure
```json
{
  "run_id": "20260130_143022",
  "config_hash": "sha256:abc123...",
  "random_seed": 42,
  "data_provenance": {
    "pxweb_snapshot_timestamp": "2026-01-29T12:00:00Z",
    "fields": [
      {"field_name": "sex", "source_type": "PXWEB_CACHED", "cache_age_days": 5}
    ]
  },
  "ipf_metrics": {...},
  "quality_gates": {...}
}
```

---

## 6. Zero-Fabrication Policy

See: [`docs/ZERO_FABRICATION_POLICY.md`](ZERO_FABRICATION_POLICY.md)

### Summary
- ✅ **Allowed**: Statistical sampling, generic occupations, real cities, cultural patterns
- ❌ **Forbidden**: Real company names, specific addresses, real person names, contact info
- **Default**: Empty string when data unavailable (never fabricated)

### Narrative Constraints
- No named employers (use "a financial institution")
- No specific addresses (use "a central neighborhood")
- No real people (politicians, celebrities)
- Romanian diacritics required (reduces hallucination)

---

## 7. Testing Matrix

| Test Type | Count | Speed | Network | Command |
|-----------|-------|-------|---------|---------|
| Unit tests | 118 | Fast | No | `pytest tests/ -m "not slow and not network"` |
| Contract tests | 11 | Fast | No | `pytest tests/test_contract.py -m "not network"` |
| API compatibility | 23 | Fast | No | `pytest tests/test_api_compatibility.py` |
| Gradient payload | 26 | Fast | No | `pytest tests/test_gradient_payload.py` |
| Network tests | 3 | Slow | Yes | `pytest tests/test_contract.py -m network` |
| **Total** | **181** | ~52s | Varies | `pytest` (full) |

### Test Markers
```python
@pytest.mark.slow      # Exclude with -m "not slow"
@pytest.mark.network   # Exclude with -m "not network"
```

### CI Strategy
- **Fast CI**: Unit + contract tests only (~10s)
- **Nightly**: + network tests + benchmarks
- **Release**: All tests + build verification

---

## 8. Operational Playbook

### When PxWeb Schema Drifts

**Symptoms:**
- `test_pxweb_sex_distribution_parsing` fails
- `ValueError: Dimension 'Sexe' not found`
- Category label mismatches

**Response:**
1. Check NBS PxWeb release notes
2. Update `CODE_MAPPINGS` in `pxweb_fetcher.py`
3. Update golden fixtures in `tests/fixtures/pxweb/`
4. Run contract tests: `pytest tests/test_contract.py -v`
5. Update `CategoryMapping` manifest
6. Bump minor version (schema change = API change)

### When Narrative Contract Breaks

**Symptoms:**
- `NarrativeContractValidator` reports missing sections
- Parse failure ratio > 10%
- Schema valid ratio < 90%

**Response:**
1. Check LLM provider changelog (model deprecation?)
2. Run regression test: `pytest tests/test_narrative_contract.py`
3. Update prompt version in `prompts.py` if needed
4. Update `NARRATIVE_CONTRACT_VERSION`
5. Retrain/update parsing logic in `parse_narrative_response()`
6. Update golden fixtures

### When ESS Drops

**Symptoms:**
- `ess_ratio < 0.70` (NARRATIVE_REQUIRED threshold)
- `weight_concentration > 5.0` (high concentration)

**Response:**
1. Increase oversample factor: `oversample_factor=5`
2. Check ethnicity distribution alignment
3. Consider accepting higher information loss for this run
4. Document in run manifest: `"information_loss_accepted": true`
5. If persistent: Review `ETHNICITY_BY_REGION` cross-tabulation

---

## 9. Versioning & Compatibility

### Semantic Versioning

| Version Change | Trigger |
|----------------|---------|
| **Major (X.0.0)** | Breaking API change, census year update, trust gate threshold changes |
| **Minor (x.Y.0)** | New features, new distributions, prompt version updates, schema drift fixes |
| **Patch (x.y.Z)** | Bug fixes, documentation, performance improvements |

### Config Hash Behavior
- Full SHA256 hash of configuration dict
- Stored in `run_manifest.config_hash`
- First 16 chars shown in summaries
- Use to verify reproducibility: same config → same hash

### Data Reference Year Handling
- `census_reference_year`: Year of hardcoded fallback data (2024)
- `data_reference_year`: Year of PxWeb data (usually same)
- **Policy**: Update only when NBS releases new census (every 10 years)
- **Migration**: Old runs remain valid; new runs use new reference

---

## 10. Quick Start (Production)

```python
from moldova_personas import PersonaGenerator, TrustReportGenerator
from moldova_personas.trust_report import UseCaseProfile

# Generate with trust gating
generator = PersonaGenerator(seed=42)
personas = generator.generate_with_ethnicity_correction(
    n=10000, 
    return_metrics=True
)

# Validate
report_gen = TrustReportGenerator(use_case=UseCaseProfile.HIGH_STAKES)
report = report_gen.generate_report(
    personas,
    random_seed=42,
    generator_version="1.0.0"
)

if report.hard_gate_triggered:
    raise ValueError(f"Quality gates failed: {report.gate_reasons}")

# Export
from moldova_personas import export_all_formats
results = export_all_formats(personas, "./output")
print(f"Generated: {results}")
print(report.summary())
```

---

## 11. Support & Issues

- **Bug Reports**: GitHub Issues with `run_manifest.json` attached
- **Schema Drift**: Tag with `pxweb-schema-drift`
- **Security**: Email security@example.com (zero-fabrication violations)

---

**Maintainer:** Moldova Personas Team  
**License:** MIT  
**Status:** ✅ Production Ready (v1.0.0)
