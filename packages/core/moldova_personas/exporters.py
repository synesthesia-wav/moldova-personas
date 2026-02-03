"""
Export utilities for persona datasets.

Supports:
- Parquet format (primary)
- JSON/JSONL
- CSV
- Statistics reports
"""

import json
from typing import List, Dict, Any, Optional, Iterable
from pathlib import Path
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .models import Persona, PersonaStatistics


class ParquetExporter:
    """Export personas to Apache Parquet format."""
    
    def __init__(self, compression: str = "snappy"):
        """
        Initialize exporter.
        
        Args:
            compression: Compression codec (snappy, gzip, brotli, etc.)
        """
        self.compression = compression
    
    def export(self, personas: List[Persona], filepath: str, drop_fields: Optional[List[str]] = None) -> None:
        """
        Export personas to Parquet file.
        
        Args:
            personas: List of personas to export
            filepath: Output file path
            drop_fields: Optional list of fields to exclude
        """
        drop_fields = set(drop_fields or [])
        # Convert to pandas DataFrame
        data = [p.model_dump(exclude=drop_fields) for p in personas]
        df = pd.DataFrame(data)
        
        # Convert list columns to strings for Parquet compatibility
        for col in ['skills_and_expertise_list', 'hobbies_and_interests_list']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else "[]")
        
        # Write to Parquet
        df.to_parquet(
            filepath,
            compression=self.compression,
            engine='pyarrow',
            index=False
        )
    
    def export_partitioned(self, personas: List[Persona], 
                          base_path: str,
                          partition_cols: Optional[List[str]] = None,
                          drop_fields: Optional[List[str]] = None) -> None:
        """
        Export with Hive-style partitioning.
        
        Args:
            personas: List of personas
            base_path: Base directory for output
            partition_cols: Columns to partition by (e.g., ["region", "sex"])
            drop_fields: Optional list of fields to exclude
        """
        drop_fields = set(drop_fields or [])
        partition_cols = partition_cols or ["region"]
        
        data = [p.model_dump(exclude=drop_fields) for p in personas]
        df = pd.DataFrame(data)
        
        # Convert list columns
        for col in ['skills_and_expertise_list', 'hobbies_and_interests_list']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else "[]")
        
        # Write partitioned
        df.to_parquet(
            base_path,
            partition_cols=partition_cols,
            compression=self.compression,
            engine='pyarrow',
            index=False
        )


class JSONExporter:
    """Export personas to JSON/JSONL format."""
    
    def export_json(self, personas: List[Persona], filepath: str, indent: int = 2, drop_fields: Optional[List[str]] = None) -> None:
        """
        Export as single JSON array.
        
        Args:
            personas: List of personas
            filepath: Output file path
            indent: JSON indentation
            drop_fields: Optional list of fields to exclude
        """
        drop_fields = set(drop_fields or [])
        data = [p.model_dump(exclude=drop_fields) for p in personas]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    def export_jsonl(self, personas: List[Persona], filepath: str, drop_fields: Optional[List[str]] = None) -> None:
        """
        Export as JSON Lines (one JSON object per line).
        
        Args:
            personas: List of personas
            filepath: Output file path
            drop_fields: Optional list of fields to exclude
        """
        drop_fields = set(drop_fields or [])
        with open(filepath, 'w', encoding='utf-8') as f:
            for persona in personas:
                f.write(json.dumps(persona.model_dump(exclude=drop_fields), ensure_ascii=False) + '\n')


class CSVExporter:
    """Export personas to CSV format."""
    
    def export(self, personas: List[Persona], filepath: str, 
               separator: str = ',', drop_fields: Optional[List[str]] = None) -> None:
        """
        Export to CSV.
        
        Args:
            personas: List of personas
            filepath: Output file path
            separator: Field separator
            drop_fields: Optional list of fields to exclude
        """
        drop_fields = set(drop_fields or [])
        data = [p.model_dump(exclude=drop_fields) for p in personas]
        df = pd.DataFrame(data)
        
        # Convert list columns to strings
        for col in ['skills_and_expertise_list', 'hobbies_and_interests_list']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: '|'.join(x) if x else '')
        
        df.to_csv(filepath, sep=separator, index=False, encoding='utf-8')


class StatisticsExporter:
    """Generate and export dataset statistics."""
    
    def generate(self, personas: List[Persona]) -> PersonaStatistics:
        """
        Generate statistics for a persona dataset.
        
        Args:
            personas: List of personas
        
        Returns:
            PersonaStatistics object
        """
        n = len(personas)
        
        # Calculate distributions
        from collections import Counter
        
        sex_dist = dict(Counter(p.sex for p in personas))
        sex_dist = {k: v/n for k, v in sex_dist.items()}
        
        age_dist = dict(Counter(p.age_group for p in personas))
        age_dist = {k: v/n for k, v in age_dist.items()}
        
        region_dist = dict(Counter(p.region for p in personas))
        region_dist = {k: v/n for k, v in region_dist.items()}
        
        ethnicity_dist = dict(Counter(p.ethnicity for p in personas))
        ethnicity_dist = {k: v/n for k, v in ethnicity_dist.items()}
        
        education_dist = dict(Counter(p.education_level for p in personas))
        education_dist = {k: v/n for k, v in education_dist.items()}
        
        marital_dist = dict(Counter(p.marital_status for p in personas))
        marital_dist = {k: v/n for k, v in marital_dist.items()}
        
        urban_rural_dist = dict(Counter(p.residence_type for p in personas))
        urban_rural_dist = {k: v/n for k, v in urban_rural_dist.items()}
        
        employment_dist = dict(Counter(p.employment_status for p in personas))
        employment_dist = {k: v/n for k, v in employment_dist.items()}
        
        return PersonaStatistics(
            total_count=n,
            sex_distribution=sex_dist,
            age_distribution=age_dist,
            region_distribution=region_dist,
            ethnicity_distribution=ethnicity_dist,
            education_distribution=education_dist,
            marital_status_distribution=marital_dist,
            urban_rural_distribution=urban_rural_dist,
            employment_status_distribution=employment_dist,
            validation_errors=0,
            validation_warnings=0
        )

    def export_json(self, stats: PersonaStatistics, filepath: str) -> None:
        """Export statistics to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats.model_dump(), f, indent=2, ensure_ascii=False)

    def export_markdown(self, stats: PersonaStatistics, filepath: str,
                       census_data=None) -> None:
        """
        Export statistics as Markdown report.
        
        Args:
            stats: Statistics to export
            filepath: Output path
            census_data: Optional census data for comparison
        """
        lines = [
            "# Moldova Personas Dataset Statistics",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total personas: {stats.total_count}",
            "",
            "## Demographic Distributions",
            "",
            "### Sex Distribution",
            "| Category | Proportion |",
            "|----------|------------|",
        ]
        
        for cat, prop in sorted(stats.sex_distribution.items()):
            lines.append(f"| {cat} | {prop:.3f} |")
        
        lines.extend([
            "",
            "### Age Distribution",
            "| Category | Proportion |",
            "|----------|------------|",
        ])
        
        for cat, prop in sorted(stats.age_distribution.items()):
            lines.append(f"| {cat} | {prop:.3f} |")
        
        lines.extend([
            "",
            "### Region Distribution",
            "| Category | Proportion |",
            "|----------|------------|",
        ])
        
        for cat, prop in sorted(stats.region_distribution.items()):
            lines.append(f"| {cat} | {prop:.3f} |")
        
        lines.extend([
            "",
            "### Ethnicity Distribution",
            "| Category | Proportion |",
            "|----------|------------|",
        ])
        
        for cat, prop in sorted(stats.ethnicity_distribution.items(), 
                               key=lambda x: x[1], reverse=True):
            lines.append(f"| {cat} | {prop:.3f} |")
        
        lines.extend([
            "",
            "### Education Distribution",
            "| Category | Proportion |",
            "|----------|------------|",
        ])
        
        for cat, prop in sorted(stats.education_distribution.items()):
            lines.append(f"| {cat} | {prop:.3f} |")
        
        lines.extend([
            "",
            "### Urban/Rural Distribution",
            "| Category | Proportion |",
            "|----------|------------|",
        ])
        
        for cat, prop in sorted(stats.urban_rural_distribution.items()):
            lines.append(f"| {cat} | {prop:.3f} |")
        
        # Add comparison to census if available
        if census_data:
            lines.extend([
                "",
                "## Comparison to 2024 Census",
                "",
                "### Sex Distribution (Dataset vs Census)",
                "| Category | Dataset | Census | Difference |",
                "|----------|---------|--------|------------|",
            ])
            
            for cat in sorted(set(list(stats.sex_distribution.keys()) + 
                                 list(census_data.SEX_DISTRIBUTION.keys()))):
                dataset_val = stats.sex_distribution.get(cat, 0)
                census_val = census_data.SEX_DISTRIBUTION.get(cat, 0)
                diff = dataset_val - census_val
                lines.append(f"| {cat} | {dataset_val:.3f} | {census_val:.3f} | {diff:+.3f} |")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def build_statistics_from_counts(
    total: int,
    sex_counts: Dict[str, int],
    age_counts: Dict[str, int],
    region_counts: Dict[str, int],
    ethnicity_counts: Dict[str, int],
    education_counts: Dict[str, int],
    marital_counts: Dict[str, int],
    residence_counts: Dict[str, int],
    employment_counts: Optional[Dict[str, int]] = None,
) -> PersonaStatistics:
    """Build PersonaStatistics from pre-aggregated counters."""
    def _normalize(counts: Dict[str, int]) -> Dict[str, float]:
        if total <= 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    employment_counts = employment_counts or {}
    return PersonaStatistics(
        total_count=total,
        sex_distribution=_normalize(sex_counts),
        age_distribution=_normalize(age_counts),
        region_distribution=_normalize(region_counts),
        ethnicity_distribution=_normalize(ethnicity_counts),
        education_distribution=_normalize(education_counts),
        marital_status_distribution=_normalize(marital_counts),
        urban_rural_distribution=_normalize(residence_counts),
        employment_status_distribution=_normalize(employment_counts) if employment_counts else None,
        validation_errors=0,
        validation_warnings=0,
    )


def export_formats(
    personas: List[Persona],
    output_dir: str,
    basename: str = "moldova_personas",
    formats: Optional[Iterable[str]] = None,
    drop_fields: Optional[Iterable[str]] = None,
    partition_cols: Optional[List[str]] = None,
    parquet_compression: str = "snappy",
    include_stats: bool = False,
    census_data=None,
) -> Dict[str, str]:
    """
    Export personas to selected formats.

    Args:
        personas: List of personas
        output_dir: Output directory
        basename: Base filename (without extension)
        formats: Iterable of formats ("parquet", "json", "jsonl", "csv")
        drop_fields: Optional fields to exclude from export
        partition_cols: Optional partition columns for Parquet
        parquet_compression: Parquet compression codec
        include_stats: Whether to export stats files
        census_data: CensusDistributions instance for markdown stats

    Returns:
        Dict mapping format name to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    formats = [f.strip().lower() for f in (formats or [])]
    results: Dict[str, str] = {}

    drop_fields_set = set(drop_fields or [])

    if "parquet" in formats:
        parquet_exporter = ParquetExporter(compression=parquet_compression)
        if partition_cols:
            parquet_dir = output_path / f"{basename}_parquet"
            parquet_exporter.export_partitioned(
                personas,
                str(parquet_dir),
                partition_cols=partition_cols,
                drop_fields=list(drop_fields_set),
            )
            results["parquet"] = str(parquet_dir)
        else:
            parquet_path = output_path / f"{basename}.parquet"
            parquet_exporter.export(
                personas,
                str(parquet_path),
                drop_fields=list(drop_fields_set),
            )
            results["parquet"] = str(parquet_path)

    if "json" in formats:
        json_path = output_path / f"{basename}.json"
        JSONExporter().export_json(personas, str(json_path), drop_fields=list(drop_fields_set))
        results["json"] = str(json_path)

    if "jsonl" in formats:
        jsonl_path = output_path / f"{basename}.jsonl"
        JSONExporter().export_jsonl(personas, str(jsonl_path), drop_fields=list(drop_fields_set))
        results["jsonl"] = str(jsonl_path)

    if "csv" in formats:
        csv_path = output_path / f"{basename}.csv"
        CSVExporter().export(personas, str(csv_path), drop_fields=list(drop_fields_set))
        results["csv"] = str(csv_path)

    if include_stats:
        stats = StatisticsExporter().generate(personas)

        stats_json_path = output_path / f"{basename}_stats.json"
        StatisticsExporter().export_json(stats, str(stats_json_path))
        results["stats_json"] = str(stats_json_path)

        stats_md_path = output_path / f"{basename}_stats.md"
        if census_data is None:
            from .census_data import CENSUS
            census_data = CENSUS
        StatisticsExporter().export_markdown(stats, str(stats_md_path), census_data=census_data)
        results["stats_markdown"] = str(stats_md_path)

    return results


def export_all_formats(
    personas: List[Persona],
    output_dir: str,
    basename: str = "moldova_personas",
) -> Dict[str, str]:
    """
    Export personas to all supported formats.

    Args:
        personas: List of personas
        output_dir: Output directory
        basename: Base filename (without extension)

    Returns:
        Dict mapping format name to file path
    """
    return export_formats(
        personas=personas,
        output_dir=output_dir,
        basename=basename,
        formats=["parquet", "json", "jsonl", "csv"],
        include_stats=True,
    )
