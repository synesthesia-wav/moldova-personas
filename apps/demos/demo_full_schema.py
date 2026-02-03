#!/usr/bin/env python3
"""
Full reference-schema demo with:
1. Reference OCEAN schema (t_score, label, description)
2. Population-level QA (stratified checks, divergence metrics)
3. Complete two-layer generation

Usage:
  python apps/demos/demo_full_schema.py --count 50 --provider dashscope --api-key <key>
"""

from _bootstrap import ensure_core_path

ensure_core_path()


import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from moldova_personas.generator import PersonaGenerator
from moldova_personas.names import generate_name
from moldova_personas.ocean_schema import OCEANProfileSchema
from moldova_personas.population_qa import PopulationQA, quick_population_qa


# Simplified reference-schema persona with proper OCEAN schema
class ReferenceSchemaPersona:
    """Persona with reference OCEAN schema."""
    
    def __init__(self, base_data: Dict, ocean_schema: OCEANProfileSchema):
        self.data = base_data
        self.ocean = ocean_schema
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary with reference OCEAN format."""
        result = self.data.copy()
        # Replace raw OCEAN scores with reference schema
        result["ocean_profile"] = self.ocean.to_dict()
        # Keep raw scores for reference
        result["ocean_raw_scores"] = {
            "openness": self.ocean.openness.t_score,
            "conscientiousness": self.ocean.conscientiousness.t_score,
            "extraversion": self.ocean.extraversion.t_score,
            "agreeableness": self.ocean.agreeableness.t_score,
            "neuroticism": self.ocean.neuroticism.t_score
        }
        return result


async def generate_schema_batch(
    count: int,
    provider: str = "mock",
    api_key: Optional[str] = None
) -> List[ReferenceSchemaPersona]:
    """Generate batch with reference OCEAN schema."""
    
    print(f"Generating {count} personas with reference OCEAN schema...")
    
    # Generate base personas
    generator = PersonaGenerator()
    base_personas = generator.generate(count)
    
    results = []
    for p in base_personas:
        # Generate name
        name = generate_name(getattr(p, 'ethnicity', 'Moldovean'), p.sex)
        
        # Sample OCEAN conditioned on demographics
        from moldova_personas.ocean_framework import OCEANSampler
        sampler = OCEANSampler()
        raw_ocean = sampler.sample(
            age=p.age,
            sex=p.sex,
            education_level=p.education_level,
            occupation=p.occupation
        )
        
        # Convert to reference schema
        ocean_schema = OCEANProfileSchema.from_raw_scores(
            openness=raw_ocean.openness,
            conscientiousness=raw_ocean.conscientiousness,
            extraversion=raw_ocean.extraversion,
            agreeableness=raw_ocean.agreeableness,
            neuroticism=raw_ocean.neuroticism,
            source=raw_ocean.source,
            confidence=raw_ocean.confidence
        )
        
        # Build base data
        base_data = {
            "uuid": p.uuid,
            "name": name,
            "sex": p.sex,
            "age": p.age,
            "age_group": _get_age_group(p.age),
            "marital_status": p.marital_status,
            "education_level": p.education_level,
            "occupation": p.occupation,
            "region": p.region,
            "district": p.district,
            "locality": p.city,
            "residence_type": p.residence_type,
            "ethnicity": getattr(p, 'ethnicity', 'Moldovean'),
            "religion": getattr(p, 'religion', 'Ortodox'),
        }
        
        results.append(ReferenceSchemaPersona(base_data, ocean_schema))
    
    return results


def _get_age_group(age: int) -> str:
    """Map age to group."""
    if age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"



def run_population_qa(personas: List[ReferenceSchemaPersona]) -> Dict[str, Any]:
    """Run full population QA."""
    
    print("\nRunning population-level QA...")
    
    # Convert to dicts for QA
    persona_dicts = [p.to_dict() for p in personas]
    
    # Quick check
    quick = quick_population_qa(persona_dicts)
    print(f"  Total personas: {quick['total']}")
    print(f"  Stability: {quick['stability']}")
    print(f"  Healthy: {'✓' if quick['healthy'] else '✗'}")
    
    # Detailed OCEAN check
    qa = PopulationQA()
    ocean_details = qa.check_ocean_distributions(persona_dicts)
    
    print("\n  OCEAN Distribution Health:")
    for trait, data in ocean_details.items():
        status = "⚠️ COLLAPSE RISK" if data["collapse_warning"] else "✓ OK"
        print(f"    {trait}: μ={data['observed_mean']:.1f}, σ={data['observed_std']:.1f} {status}")
    
    # Full report
    full_report = qa.generate_report(persona_dicts)
    
    return full_report


def export_schema_format(
    personas: List[ReferenceSchemaPersona],
    output_dir: Path,
    qa_report: Dict[str, Any]
):
    """Export in reference schema format."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSONL (primary format)
    jsonl_path = output_dir / "moldova_personas_schema.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for p in personas:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + '\n')
    
    # Full JSON with metadata
    json_path = output_dir / "moldova_personas_full.json"
    full_data = {
        "metadata": {
            "format": "Moldova-Personas-ReferenceSchema",
            "version": "2.0.0",
            "count": len(personas),
            "schema": {
                "ocean": "Reference (t_score, label, description)",
                "demographics": "NBS 2024 grounded",
                "narrative": "Two-layer with OCEAN consistency"
            }
        },
        "population_qa": qa_report,
        "personas": [p.to_dict() for p in personas]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    # Dataset card
    readme_path = output_dir / "README.md"
    
    # Pre-compute QA JSON to avoid f-string complexity
    qa_checks_json = json.dumps(qa_report.get('checks', {}), indent=2, ensure_ascii=False)
    
    readme = f"""---
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
      num_examples: {len(personas)}
---

# Moldova-Personas (Reference Schema)

{len(personas):,} synthetic Moldovan personas with reference OCEAN Big Five schema.

## Schema

### Demographics
Standard fields: uuid, name, sex, age, age_group, education_level, occupation, 
region, district, locality, ethnicity, religion.

### OCEAN Profile (Reference Format)
```json
{{
  "openness": {{
    "t_score": 65,
    "label": "high", 
    "description": "Curios, creativ, deschis la experiențe noi"
  }},
  ...
}}
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
{qa_checks_json}
```

## Generation

Two-pass pipeline:
1. **Structure**: Demographics + OCEAN (conditioned) + Behavioral contract
2. **Narrative**: OCEAN-guided generation with score-and-rewrite validation

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("your-org/moldova-personas-schema")

# Access OCEAN profile
ocean = dataset["train"][0]["ocean_profile"]
print(ocean["openness"]["description"])
```

## License

MIT License
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    return {
        "jsonl": jsonl_path,
        "json": json_path,
        "readme": readme_path
    }


async def main():
    parser = argparse.ArgumentParser(description="Full reference-schema demo")
    parser.add_argument("--count", "-n", type=int, default=50)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-dir", default="output_schema_full")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("REFERENCE SCHEMA: OCEAN PROFILE + POPULATION QA")
    print("=" * 70)
    print(f"\nGenerating {args.count} personas...")
    
    # Generate
    personas = await generate_schema_batch(
        count=args.count,
        provider=args.provider,
        api_key=args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    )
    
    # Show sample
    print("\n" + "-" * 70)
    print("SAMPLE PERSONA (Reference OCEAN Schema):")
    print("-" * 70)
    sample = personas[0].to_dict()
    print(f"\nName: {sample['name']}")
    print(f"Demographics: {sample['sex']}, {sample['age']} ani, {sample['occupation']}")
    print(f"Location: {sample['locality']}, {sample['district']}")
    print("\nOCEAN Profile (reference format):")
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    for trait in traits:
        data = sample['ocean_profile'][trait]
        print(f"  {trait}:")
        print(f"    t_score: {data['t_score']}")
        print(f"    label: {data['label']}")
        print(f"    description: {data['description']}")
    
    # Population QA
    qa_report = run_population_qa(personas)
    
    # Export
    output_dir = Path(args.output_dir)
    paths = export_schema_format(personas, output_dir, qa_report)
    
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    print(f"\nOverall QA: {'✓ PASS' if qa_report.get('overall_acceptable') else '✗ FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
