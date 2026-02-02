"""
Three explicit export formats for Nemotron personas.

1. internal_full.jsonl - Complete with internals for debugging
2. nemotron_compat.jsonl - Exact Nemotron-Brazil field set
3. moldova_extended_public.jsonl - Extended public fields
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# Nemotron-Brazil public schema (exact field set)
NEMOTRON_BRAZIL_FIELDS = {
    "uuid",
    "persona",
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
    "municipality",
    "state",
    "country",
}

# Internal fields (never export to public)
INTERNAL_FIELDS = {
    "ocean_scores",
    "ocean_raw_scores",
    "behavioral_cues",
    "constraints",
    "validation",
    "passed_validation",
    "rewrite_count",
    "ocean_deviation_score",
    "repair_history",
    "prompt_hash",
    "trace_id",
    "generation_metadata",
}

# Moldova extended fields (public but beyond Nemotron schema)
MOLDOVA_EXTENDED_FIELDS = {
    "locality",
    "ethnicity",
    "religion",
    "residence_type",
    "age_group",
    "ocean_profile",
}


@dataclass
class ExportConfig:
    """Configuration for export."""
    schema_version: str = "2.0.0"
    generator_version: str = "1.0.0"
    model_name: str = "qwen-turbo"
    provider: str = "dashscope"
    generation_date: str = ""
    seed: Optional[int] = None


class ExportManager:
    """
    Manage three export formats for Nemotron personas.
    
    Usage:
        exporter = ExportManager(config)
        
        # Export all formats
        paths = exporter.export_all(personas, output_dir)
        
        # Or individual formats
        exporter.export_internal(personas, output_dir / "internal.jsonl")
        exporter.export_nemotron_compat(personas, output_dir / "nemotron.jsonl")
        exporter.export_moldova_extended(personas, output_dir / "extended.jsonl")
    """
    
    def __init__(self, config: ExportConfig):
        self.config = config
    
    def export_all(
        self,
        personas: List[Dict],
        output_dir: Path,
        include_quality_report: bool = True
    ) -> Dict[str, Path]:
        """
        Export all three formats.
        
        Returns:
            Dict mapping format name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. Internal full format
        results["internal"] = self.export_internal(
            personas,
            output_dir / "internal_full.jsonl"
        )
        
        # 2. Nemotron-compatible format
        results["nemotron"] = self.export_nemotron_compat(
            personas,
            output_dir / "nemotron_compat.jsonl"
        )
        
        # 3. Moldova extended format
        results["extended"] = self.export_moldova_extended(
            personas,
            output_dir / "moldova_extended_public.jsonl"
        )
        
        # 4. Metadata file
        results["metadata"] = self._export_metadata(
            personas,
            output_dir / "metadata.json"
        )
        
        # 5. Quality report (optional)
        if include_quality_report:
            results["quality"] = self._export_quality_report(
                personas,
                output_dir / "quality_report.json"
            )
        
        return results
    
    def export_internal(
        self,
        personas: List[Dict],
        output_path: Path
    ) -> Path:
        """
        Export internal full format with all fields.
        
        Includes: ocean_scores, behavioral_cues, constraints, validation, etc.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for persona in personas:
                # Add export metadata
                record = {
                    **persona,
                    "_export_metadata": {
                        "schema_version": self.config.schema_version,
                        "generator_version": self.config.generator_version,
                        "export_timestamp": self.config.generation_date,
                        "format": "internal_full"
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return output_path
    
    def export_nemotron_compat(
        self,
        personas: List[Dict],
        output_path: Path
    ) -> Path:
        """
        Export Nemotron-Brazil compatible format.
        
        Exact field set matching their public schema.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for persona in personas:
                # Filter to Nemotron fields only
                record = {
                    k: v for k, v in persona.items()
                    if k in NEMOTRON_BRAZIL_FIELDS
                }
                
                # Add minimal metadata
                record["_source"] = {
                    "generator": "nemotron-moldova",
                    "version": self.config.generator_version,
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return output_path
    
    def export_moldova_extended(
        self,
        personas: List[Dict],
        output_path: Path
    ) -> Path:
        """
        Export Moldova extended public format.
        
        Includes Nemotron fields + Moldova-specific extensions.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        allowed_fields = NEMOTRON_BRAZIL_FIELDS | MOLDOVA_EXTENDED_FIELDS
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for persona in personas:
                # Filter to public fields only
                record = {
                    k: v for k, v in persona.items()
                    if k in allowed_fields and k not in INTERNAL_FIELDS
                }
                
                # Ensure OCEAN is in NeMo format
                if "ocean_profile" in persona:
                    record["ocean_profile"] = persona["ocean_profile"]
                
                # Add metadata
                record["_source"] = {
                    "generator": "nemotron-moldova",
                    "version": self.config.generator_version,
                    "schema": "moldova_extended_public",
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return output_path
    
    def _export_metadata(
        self,
        personas: List[Dict],
        output_path: Path
    ) -> Path:
        """Export generation metadata."""
        metadata = {
            "schema_version": self.config.schema_version,
            "generator_version": self.config.generator_version,
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "generation_date": self.config.generation_date,
            "seed": self.config.seed,
            "record_count": len(personas),
            "formats": {
                "internal_full": {
                    "description": "Complete record with internals for debugging",
                    "fields": "all",
                },
                "nemotron_compat": {
                    "description": "Exact Nemotron-Brazil field compatibility",
                    "fields": sorted(NEMOTRON_BRAZIL_FIELDS),
                },
                "moldova_extended_public": {
                    "description": "Nemotron schema + Moldova-specific fields",
                    "fields": sorted(NEMOTRON_BRAZIL_FIELDS | MOLDOVA_EXTENDED_FIELDS),
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _export_quality_report(
        self,
        personas: List[Dict],
        output_path: Path
    ) -> Path:
        """Export quality metrics report."""
        from .quality_dashboard import QualityDashboard
        
        dashboard = QualityDashboard()
        metrics = dashboard.analyze_dataset(personas)
        
        report = {
            "timestamp": metrics.timestamp,
            "total_records": metrics.total_records,
            "quality_score": self._calculate_quality_score(metrics),
            "metrics": metrics.to_dict(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _calculate_quality_score(self, metrics) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Deduct for issues
        if metrics.trigram_repetition_rate > 0.1:
            score -= 10
        if metrics.rare_anchor_compliance_rate < 0.5:
            score -= 10
        if metrics.anchor_overuse_violations:
            score -= len(metrics.anchor_overuse_violations) * 5
        if metrics.estimated_uniqueness_ratio < 0.9:
            score -= 10
        
        return max(0.0, score)


def validate_export_compliance(
    file_path: Path,
    expected_format: str
) -> Dict[str, Any]:
    """
    Validate that an export file complies with expected format.
    
    Returns:
        Validation report
    """
    issues = []
    
    if expected_format == "nemotron_compat":
        required_fields = NEMOTRON_BRAZIL_FIELDS
        forbidden_fields = INTERNAL_FIELDS | MOLDOVA_EXTENDED_FIELDS
    elif expected_format == "moldova_extended":
        required_fields = NEMOTRON_BRAZIL_FIELDS
        forbidden_fields = INTERNAL_FIELDS
    else:
        return {"valid": False, "error": f"Unknown format: {expected_format}"}
    
    # Check first few records
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Check first 5 records
                break
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"Line {i+1}: Invalid JSON - {e}")
                continue
            
            # Check required fields
            for field in required_fields:
                if field not in record:
                    issues.append(f"Line {i+1}: Missing required field '{field}'")
            
            # Check for forbidden fields
            for field in forbidden_fields:
                if field in record and not field.startswith("_"):
                    issues.append(f"Line {i+1}: Forbidden field '{field}' present")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "format": expected_format,
    }


def export_personas(
    personas: List[Dict],
    output_dir: str,
    config: Optional[ExportConfig] = None
) -> Dict[str, Path]:
    """
    Convenience function to export personas in all formats.
    
    Example:
        paths = export_personas(personas, "./output")
        print(f"Internal: {paths['internal']}")
        print(f"Nemotron: {paths['nemotron']}")
    """
    from datetime import datetime
    
    if config is None:
        config = ExportConfig(
            generation_date=datetime.now().isoformat()
        )
    
    exporter = ExportManager(config)
    return exporter.export_all(personas, Path(output_dir))
