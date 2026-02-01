---
language:
  - ro
dataset_info:
  features:
    - name: uuid
      dtype: string
    - name: name
      dtype: string
    - name: sex
      dtype: string
    - name: age
      dtype: int32
    - name: age_group
      dtype: string
    - name: occupation
      dtype: string
    - name: region
      dtype: string
    - name: ocean_profile
      dtype: struct
  splits:
    - name: train
      num_examples: 100
---

# Nemotron-Personas-Moldova (NeMo Schema)

100 synthetic Moldovan personas with NeMo-style OCEAN Big Five schema.

## Schema

### Demographics
Standard fields: uuid, name, sex, age, age_group, education_level, occupation, 
region, district, locality, ethnicity, religion.

### OCEAN Profile (NeMo Format)
```json
{
  "openness": {
    "t_score": 65,
    "label": "high", 
    "description": "Curios, creativ, deschis la experiențe noi"
  },
  ...
}
```

Each trait includes:
- `t_score`: 0-100 population-normalized score
- `label`: "low", "average", or "high"
- `description`: Human-readable trait expression

### Population QA
This dataset includes population-level quality metrics:
- OCEAN distribution divergence (KL/JS)
- Variance checks (no collapse)
- Stability across batches

## Quality Metrics

```json
{
  "ocean_distributions": {},
  "ocean_stability": {
    "drift_detected": false,
    "drifted_traits": [],
    "stable_traits": [],
    "overall_stable": true
  }
}
```

## Generation

Two-pass pipeline:
1. **Structure**: Demographics + OCEAN (conditioned) + Behavioral contract
2. **Narrative**: OCEAN-guided generation with score-and-rewrite validation

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("your-org/nemotron-personas-moldova-nemo")

# Access OCEAN profile
ocean = dataset["train"][0]["ocean_profile"]
print(ocean["openness"]["description"])
```

## License

MIT License
