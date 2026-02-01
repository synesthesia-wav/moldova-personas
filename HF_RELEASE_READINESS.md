# HuggingFace Release Readiness Checklist

## Pre-Release Verification

### ✅ Data Quality

- [ ] **Scale validation passed**
  - [ ] Tier 1 (1K): Smoke test complete
  - [ ] Tier 2 (25K): Distribution + drift check complete
  - [ ] Tier 3 (100K): Throughput + cost validation complete

- [ ] **Quality dashboard metrics acceptable**
  - [ ] Opening move distribution roughly uniform (no type >40%)
  - [ ] Trigram repetition <10%
  - [ ] Rare anchor compliance >50%
  - [ ] No anchor overuse violations (>5% share)
  - [ ] Estimated uniqueness >90%

- [ ] **Golden set regression passed**
  - [ ] 200 cores regenerated
  - [ ] Validator pass rate stable (±5%)
  - [ ] Mean edits per record stable (±0.5)
  - [ ] Duplication rate <2%
  - [ ] Q&A contradiction rate <5%

### ✅ Safety & Ethics

- [ ] **Sensitive claims gate passed**
  - [ ] No ethnicity stereotypes detected
  - [ ] No religion stereotypes detected
  - [ ] No gender stereotypes detected
  - [ ] No regional stereotypes detected

- [ ] **Protected attribute determinism checked**
  - [ ] Ethnicity-occupation correlation within bounds
  - [ ] Religion-occupation correlation within bounds
  - [ ] No sensitive occupation links detected

- [ ] **Distribution fairness verified**
  - [ ] Ethnicity distribution matches census (±15%)
  - [ ] Religion distribution matches census (±15%)

### ✅ Export Formats

- [ ] **internal_full.jsonl** generated
  - [ ] Contains all debug fields
  - [ ] Trace IDs present
  - [ ] Prompt hashes recorded

- [ ] **nemotron_compat.jsonl** generated
  - [ ] Exact Nemotron-Brazil field set
  - [ ] No internal fields leaked
  - [ ] OCEAN in NeMo format

- [ ] **moldova_extended_public.jsonl** generated
  - [ ] Nemotron fields + Moldova extensions
  - [ ] Ethnicity/religion included (if publishing)
  - [ ] No internal fields

- [ ] **metadata.json** complete
  - [ ] schema_version
  - [ ] generator_version
  - [ ] model/provider info
  - [ ] generation_date
  - [ ] seed

### ✅ Documentation

- [ ] **Dataset card (README.md) complete**
  ```yaml
  ---
  language:
    - ro
  dataset_info:
    features:
      - name: uuid
        dtype: string
      # ... all fields
  ---
  ```

- [ ] **Content sections**
  - [ ] Dataset description
  - [ ] Data sources (NBS 2024 Census)
  - [ ] Generation pipeline summary
  - [ ] Schema documentation
  - [ ] Privacy constraints (synthetic, no PII)
  - [ ] Known limitations
  - [ ] Recommended uses
  - [ ] Citation info

- [ ] **Quickstart notebook**
  - [ ] Load JSONL
  - [ ] Show example personas
  - [ ] Filter by demographics
  - [ ] Access OCEAN profile

### ✅ Reproducibility

- [ ] **Fixed seeds documented**
  - [ ] Base generation seed
  - [ ] OCEAN sampling seed
  - [ ] Golden set seed

- [ ] **Config snapshot saved**
  - [ ] All hyperparameters
  - [ ] Prompt versions
  - [ ] Validator thresholds

- [ ] **Model/provider versions**
  - [ ] LLM provider (e.g., DashScope)
  - [ ] Model name (e.g., qwen-turbo)
  - [ ] API version/date

### ✅ Publication Artifacts

```
output_release/
├── README.md                    # Dataset card
├── metadata.json                # Generation metadata
├── internal_full.jsonl          # Debug format (optional)
├── nemotron_compat.jsonl        # Nemotron-Brazil compatible
├── moldova_extended_public.jsonl # Extended public format
├── quality_report.json          # Quality metrics
├── safety_report.json           # Safety gate results
└── quickstart.ipynb             # Example usage
```

## Upload Commands

```bash
# Install HF CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Upload dataset
huggingface-cli upload your-org/nemotron-personas-moldova \
    ./output_release/*.jsonl \
    ./output_release/*.json \
    ./output_release/README.md \
    --repo-type dataset

# Upload notebook separately
huggingface-cli upload your-org/nemotron-personas-moldova \
    ./output_release/quickstart.ipynb \
    --repo-type dataset
```

## Post-Upload Verification

```python
from datasets import load_dataset

# Load and verify
dataset = load_dataset("your-org/nemotron-personas-moldova", split="train")

# Check record count
assert len(dataset) == 100000, f"Expected 100K, got {len(dataset)}"

# Check schema
required_fields = ["uuid", "persona", "ocean_profile"]
for field in required_fields:
    assert field in dataset.features, f"Missing field: {field}"

# Sample record
sample = dataset[0]
print(f"Sample OCEAN: {sample['ocean_profile']['openness']}")
```

## Versioning Strategy

| Version | Records | Changes |
|---------|---------|---------|
| v1.0.0 | 100K | Initial release |
| v1.1.0 | 100K | Prompt improvements |
| v2.0.0 | 500K | Scale increase |

Use semantic versioning:
- MAJOR: Breaking schema changes
- MINOR: New fields, improvements
- PATCH: Bug fixes, same schema

## Rollback Plan

If issues discovered post-release:

1. **Immediate**: Mark dataset as deprecated in card
2. **Within 24h**: Upload hotfix version (v1.0.1)
3. **Communication**: Update README with known issues
4. **Cleanup**: Delete/replace problematic version after fix confirmed

## Contact & Support

- Dataset issues: GitHub issues
- Usage questions: HF Discussions
- Research collaboration: Email

---

**Release approval**: [ ] Date: _______

**Released by**: _________________
