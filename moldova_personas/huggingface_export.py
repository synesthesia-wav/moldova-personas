"""
Export Nemotron-style personas to HuggingFace Datasets format.

Compatible with Nemotron-Personas-Brazil schema for easy integration
with the HuggingFace ecosystem.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .nemotron_pipeline import NemotronPersona


class HuggingFaceExporter:
    """
    Export personas to HuggingFace-compatible formats.
    
    Supports:
    - JSON Lines (.jsonl) - recommended for large datasets
    - Parquet (.parquet) - compressed columnar format
    - CSV (.csv) - for easy inspection
    - Dataset card template (.md)
    """
    
    # Nemotron-Brazil compatible field order
    FIELD_ORDER = [
        "uuid",
        "persona",  # General essence
        "professional_persona",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
        "cultural_background",
        "skills_and_expertise",
        "skills_and_expertise_list",
        "hobbies_and_interests",
        "hobbies_and_interests_list",
        "career_goals_and_ambitions",
        "sex",
        "age",
        "marital_status",
        "education_level",
        "occupation",
        "municipality",  # raion
        "state",  # region
        "country",
        "locality",  # village/city
        "ethnicity",
        "religion",
    ]
    
    def __init__(self, dataset_name: str = "moldova-personas"):
        self.dataset_name = dataset_name
    
    def to_jsonl(self, personas: List[NemotronPersona], output_path: Path) -> Path:
        """
        Export to JSON Lines format (one JSON object per line).
        
        This is the recommended format for HuggingFace datasets.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for persona in personas:
                record = self._to_ordered_dict(persona)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return output_path
    
    def to_json(self, personas: List[NemotronPersona], output_path: Path) -> Path:
        """Export to single JSON file with metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            "metadata": self._generate_metadata(len(personas)),
            "personas": [self._to_ordered_dict(p) for p in personas]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def to_csv(self, personas: List[NemotronPersona], output_path: Path) -> Path:
        """Export to CSV format."""
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if personas:
                writer = csv.DictWriter(f, fieldnames=self.FIELD_ORDER)
                writer.writeheader()
                for persona in personas:
                    writer.writerow(self._to_ordered_dict(persona))
        
        return output_path
    
    def to_parquet(self, personas: List[NemotronPersona], output_path: Path) -> Path:
        """
        Export to Parquet format (compressed columnar).
        
        Best for large datasets and ML training.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet export. Install with: pip install pandas pyarrow")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to list of dicts
        records = [self._to_ordered_dict(p) for p in personas]
        
        # Create DataFrame and save
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False, compression='snappy')
        
        return output_path
    
    def generate_dataset_card(
        self,
        output_path: Path,
        num_personas: int,
        source_census: str = "NBS 2024",
        version: str = "1.0.0"
    ) -> Path:
        """
        Generate a HuggingFace dataset card (README.md).
        
        Follows the Nemotron-Brazil card style.
        """
        card = f"""---
language:
  - ro
dataset_info:
  features:
    - name: uuid
      dtype: string
    - name: persona
      dtype: string
    - name: professional_persona
      dtype: string
    - name: sports_persona
      dtype: string
    - name: arts_persona
      dtype: string
    - name: travel_persona
      dtype: string
    - name: culinary_persona
      dtype: string
    - name: cultural_background
      dtype: string
    - name: skills_and_expertise
      dtype: string
    - name: skills_and_expertise_list
      dtype: string
    - name: hobbies_and_interests
      dtype: string
    - name: hobbies_and_interests_list
      dtype: string
    - name: career_goals_and_ambitions
      dtype: string
    - name: sex
      dtype: string
    - name: age
      dtype: int32
    - name: marital_status
      dtype: string
    - name: education_level
      dtype: string
    - name: occupation
      dtype: string
    - name: municipality
      dtype: string
    - name: state
      dtype: string
    - name: country
      dtype: string
    - name: locality
      dtype: string
    - name: ethnicity
      dtype: string
    - name: religion
      dtype: string
  splits:
    - name: train
      num_examples: {num_personas}
  config_name: default
  dataset_size: ~{num_personas * 2}MB
---

# Nemotron-Personas-Moldova

A dataset of {num_personas:,} synthetic Moldovan personas grounded in {source_census} census data.

## Overview

This dataset provides culturally authentic, demographically accurate synthetic personas for training Romanian-language AI systems. Every persona is statistically grounded in official census data but represents no real individual.

## Dataset Schema

Each record contains 24 fields across 4 categories:

### 1. Core Personas (6 short variants)
- `persona` - General essence statement
- `professional_persona` - Work context
- `sports_persona` - Sports/activities
- `arts_persona` - Cultural interests  
- `travel_persona` - Travel style
- `culinary_persona` - Food preferences

### 2. Context Fields (shared latent story)
- `cultural_background` - Regional/cultural context
- `skills_and_expertise` + `skills_and_expertise_list`
- `hobbies_and_interests` + `hobbies_and_interests_list`
- `career_goals_and_ambitions`

### 3. Demographics
- `sex`, `age`, `marital_status`
- `education_level`, `occupation`
- `ethnicity`, `religion`

### 4. Geography
- `country` = "Moldova"
- `state` = Region (Chisinau, Nord, Centru, Sud, Gagauzia)
- `municipality` = Raion (district)
- `locality` = City/village

## Generation Method

**Stage A**: Structured core via IPF on census microdata  
**Stage B**: Context fields via LLM (Qwen-Turbo)  
**Stage C**: 6 persona variants derived from context  
**Validation**: Hard validators + LLM-as-judge soft validation

## Quality Metrics

- Demographic accuracy: Matched to NBS 2024 distributions
- Cultural authenticity: Romanian language, Moldovan context
- Coherence: All 6 persona variants share same latent story
- Privacy: No real PII, fully synthetic

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("your-org/nemotron-personas-moldova")

# Access a persona
persona = dataset["train"][0]
print(persona["persona"])
print(persona["cultural_background"])
```

## Applications

- Fine-tuning Romanian LLMs
- Cultural alignment testing
- Demographic bias evaluation
- Synthetic dialogue generation

## License

MIT License - Commercial use allowed

## Citation

```bibtex
@dataset{{nemotron_personas_moldova_{version},
  title = {{Nemotron-Personas-Moldova: Synthetic Personas Grounded in Census Data}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  version = {{{version}}}
}}
```
"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(card)
        
        return output_path
    
    def export_all(
        self,
        personas: List[NemotronPersona],
        output_dir: Path,
        formats: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Export to all formats.
        
        Args:
            personas: List of personas to export
            output_dir: Output directory
            formats: List of formats (default: all)
            
        Returns:
            Dict mapping format name to file path
        """
        formats = formats or ["jsonl", "json", "csv", "parquet"]
        output_dir = Path(output_dir)
        
        results = {}
        base_name = self.dataset_name
        
        if "jsonl" in formats:
            results["jsonl"] = self.to_jsonl(
                personas, 
                output_dir / f"{base_name}.jsonl"
            )
        
        if "json" in formats:
            results["json"] = self.to_json(
                personas,
                output_dir / f"{base_name}.json"
            )
        
        if "csv" in formats:
            results["csv"] = self.to_csv(
                personas,
                output_dir / f"{base_name}.csv"
            )
        
        if "parquet" in formats:
            try:
                results["parquet"] = self.to_parquet(
                    personas,
                    output_dir / f"{base_name}.parquet"
                )
            except ImportError:
                pass  # Skip if pandas not installed
        
        # Always generate dataset card
        results["readme"] = self.generate_dataset_card(
            output_dir / "README.md",
            len(personas)
        )
        
        return results
    
    def _to_ordered_dict(self, persona: NemotronPersona) -> Dict[str, Any]:
        """Convert persona to ordered dict following Nemotron schema."""
        data = persona.to_dict()
        # Ensure consistent field order
        return {k: data.get(k, "") for k in self.FIELD_ORDER}
    
    def _generate_metadata(self, count: int) -> Dict[str, Any]:
        """Generate dataset metadata."""
        return {
            "name": self.dataset_name,
            "format": "Nemotron-Moldova",
            "count": count,
            "fields": len(self.FIELD_ORDER),
            "field_names": self.FIELD_ORDER,
            "language": "ro",
            "country": "Moldova",
            "version": "1.0.0"
        }


def export_to_huggingface_format(
    personas: List[NemotronPersona],
    output_dir: str,
    dataset_name: str = "moldova-personas",
    formats: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Convenience function to export personas to HuggingFace format.
    
    Args:
        personas: List of NemotronPersona objects
        output_dir: Output directory
        dataset_name: Name for the dataset
        formats: List of formats to export (default: all)
        
    Returns:
        Dict mapping format to file path
        
    Example:
        >>> from moldova_personas.huggingface_export import export_to_huggingface_format
        >>> export_to_huggingface_format(personas, "./hf_export")
    """
    exporter = HuggingFaceExporter(dataset_name)
    return exporter.export_all(personas, Path(output_dir), formats)
